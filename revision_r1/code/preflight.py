#!/usr/bin/env python
"""실험을 걸기 전에 돌리는 사전점검.

    python preflight.py              # 전체 점검
    python preflight.py --quick      # 무거운 항목 생략

이 파일이 존재하는 이유 — 실제로 발생했던 사고들:
  * 50에폭 진단 런이 같은 이름의 200에폭 baseline(78.55)을 덮어써 파괴
  * FocusedDropout 이 post-ReLU 입력에서 drop 비율 0.000 (완전한 no-op)
  * WCD 의 clamp 때문에 실효 드롭률이 nominal 의 2배
  * rescale 배율이 1/(1-p) 로 1e6 까지 뛰어 학습 붕괴
  * norm='max' 에서 대조군만 정확한 예산을 받아 어블레이션이 편향
  * SE/CBAM 이 lazy 생성이라 optimizer 가 파라미터를 못 잡아 미학습
공통점은 전부 "실제 입력 형태로 한 번만 돌려봤으면 즉시 보였다" 는 것이다.
"""
import sys, glob, os, csv, itertools
import numpy as np
import torch
import torch.nn as nn

FAIL, WARN = [], []


def check(name, cond, detail=""):
    print(f"  {'✔' if cond else '✘'} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(f"{name}: {detail}")
    return cond


def warn(name, cond, detail=""):
    if not cond:
        print(f"  ⚠ {name}   {detail}")
        WARN.append(f"{name}: {detail}")
    return cond


def realistic_act(B=32, C=256, H=8, W=8, sparsity=0.7, skew=True):
    """실제 post-ReLU 활성 흉내: 절반 이상이 정확히 0, 채널별 크기 편차 큼."""
    x = torch.randn(B, C, H, W)
    if skew:
        x = x * torch.rand(1, C, 1, 1).pow(2) * 5.0
    x = torch.relu(x)
    mask = (torch.rand_like(x) > sparsity).float()
    return x * mask


# ---------------------------------------------------------------------------
def check_budget():
    """모든 점수 계열이 nominal p 와 같은 실효 드롭률을 내는가 (norm='mean')."""
    print("\n[1] 드롭 예산 일치 — 점수 함수가 달라도 실효율이 같아야 한다")
    from sdrop import SDrop
    x = realistic_act()
    for p in (0.1, 0.3, 0.5):
        eff = {}
        for name, sc in [('egpg', 'egpg'), ('energy', 'energy'),
                         ('peak', 'peak'), ('random', 'random')]:
            d = SDrop(p, score=sc, norm='mean'); d.train(); d.reset_monitor()
            for _ in range(60):
                d(x)
            eff[name] = d.effective_drop_rate()
        spread = max(eff.values()) - min(eff.values())
        det = "  ".join(f"{k}={v:.4f}" for k, v in eff.items())
        if p <= 0.3:
            check(f"p={p}: 계열 간 실효율 편차 {spread:.4f} < 0.02", spread < 0.02, det)
        else:
            # norm='mean' 의 clamp(max=1) 이 고드롭률에서 물린다. 점수가 치우칠수록
            # 상한에 걸리는 채널이 늘어 실효율이 nominal 아래로 내려간다.
            # -> 예산 일치는 p<=0.3 에서만 성립. 고드롭률 비교는 예산 불일치다.
            warn(f"p={p}: 계열 간 실효율 편차 {spread:.4f} (clamp 로 예산 어긋남 — "
                 f"고드롭률 비교는 예산 불일치임을 논문에 명시할 것)", spread < 0.02, det)


def check_rescale():
    """재정규화가 train/eval 기댓값을 맞추고, 배율이 폭주하지 않는가."""
    print("\n[2] 재정규화 — 기댓값 일치와 배율 상한")
    from sdrop import SDrop
    x = realistic_act()
    xs = realistic_act(); xs[:, :3] *= 500.0          # 소수 채널이 독점하는 극단
    for tag, inp in [("보통 입력", x), ("치우친 입력", xs)]:
        for p in (0.1, 0.5):
            d = SDrop(p, score='egpg', norm='mean', rescale=True); d.train()
            outs = [d(inp) for _ in range(80)]
            amp = max(float(o.abs().max()) for o in outs) / float(inp.abs().max())
            tr = torch.stack([o.abs().mean() for o in outs]).mean()
            d.eval(); ev = d(inp).abs().mean()
            check(f"{tag} p={p}: 배율 {amp:.1f}x <= 11", amp <= 11.0)
            check(f"{tag} p={p}: train/eval 비 {tr/ev:.3f}", abs(tr/ev - 1) < 0.12)


def check_modules_bite():
    """모든 드롭 모듈이 실제 post-ReLU 입력에서 실제로 무언가를 떨어뜨리는가."""
    print("\n[3] 모듈이 실제 입력에서 작동하는가 (no-op 탐지)")
    from baselines import build_baseline
    # 두 가지 입력으로 잰다.
    #   dense  : 사전 0 이 없어 (y==0) 비율이 곧 드롭 비율 -> 모니터 없는 모듈용
    #   post-ReLU: 실제 삽입 지점의 분포. no-op 버그는 여기서만 드러난다.
    #     (FocusedDropout 이 `>=` 임계값 때문에 여기서 drop 0.000 이었다)
    dense = torch.ones(64, 256, 8, 8)
    real = realistic_act(B=64, C=256)
    for m in ('dropblock', 'wcd', 'focuseddrop'):
        kw = dict(participation=1.0) if m == 'focuseddrop' else {}
        for tag, x in (("dense", dense), ("post-ReLU", real)):
            b = build_baseline(m, 0.3, channels=256, **kw); b.train()
            fr = []
            for _ in range(30):
                y = b(x)
                f = getattr(b, 'last_drop_frac', float('nan'))
                if f == f:
                    fr.append(float(f))
                else:                       # 모니터가 없는 모듈: 0 이 된 비율에서
                    z0 = (x == 0).float().mean()
                    fr.append(float(((y == 0).float().mean() - z0) / (1 - z0)))
            fr = float(np.mean(fr))
            check(f"{m} [{tag}]: 실효 드롭 {fr:.4f} (nominal 0.3)", 0.15 < fr < 0.45)


def check_params_before_forward():
    """optimizer 생성 시점(=첫 forward 이전)에 모든 파라미터가 존재하는가."""
    print("\n[4] 지연 생성 파라미터 — optimizer 가 놓치는 가중치가 없는가")
    from model import build_model
    for arch in ('resnet18', 'resnet56'):
        for m in ('senet', 'cbam', 'sdrop', 'none'):
            net = build_model(arch, 100, m, 0.1, ['L4'])
            before = sum(p.numel() for p in net.parameters())
            net(torch.randn(2, 3, 32, 32))
            after = sum(p.numel() for p in net.parameters())
            check(f"{arch}/{m}: 첫 forward 전후 파라미터 동일", before == after,
                  f"{before:,} -> {after:,}")


def check_smoke():
    """모든 method × 대표 백본이 build->forward->backward->step 을 통과하는가."""
    print("\n[5] 스모크 — 전 method 가 학습 1스텝을 통과하는가")
    from model import build_model
    METHODS = ['none', 'dropout', 'dropout_std', 'spatialdropout', 'sdrop',
               'sdrop_energy', 'sdrop_peak', 'sdrop_random', 'sgridlc',
               'dropblock', 'senet', 'cbam', 'wcd', 'focuseddrop']
    for arch, layers in [('resnet18', ['L4']), ('resnet56', ['L3'])]:
        bad = []
        for m in METHODS:
            try:
                net = build_model(arch, 100, m, 0.1, layers)
                opt = torch.optim.SGD(net.parameters(), lr=0.01)
                y = net(torch.randn(4, 3, 32, 32))
                loss = nn.functional.cross_entropy(y, torch.randint(0, 100, (4,)))
                loss.backward(); opt.step()
                if not torch.isfinite(loss):
                    bad.append(f"{m}(loss={loss})")
            except Exception as e:
                bad.append(f"{m}({type(e).__name__}: {str(e)[:40]})")
        check(f"{arch}: {len(METHODS)}개 method 전부 통과", not bad, "; ".join(bad))


def check_insertion_activation():
    """삽입 지점의 입력이 post-ReLU 인가.

    같은 --layers 플래그가 아키텍처마다 다른 것을 재면 부호 일관성 격자가
    무효가 된다. WRN(pre-activation)의 L2/L3 는 음수 52% 였다.
    """
    print("\n[8] 삽입 지점 입력이 post-ReLU 인가")
    from model import build_model
    for arch, layers in [('resnet56', ['L2', 'L3', 'L4']), ('wrn28_10', ['L4'])]:
        m = build_model(arch, 100, 'sdrop', 0.1, layers)
        m.eval()
        seen = {}

        def hook(name):
            def f(mod, inp, out):
                seen[name] = float((inp[0] < 0).float().mean())
            return f

        hs = [d.register_forward_hook(hook(nm))
              for nm, d in zip(m.stage_names, m.drops)
              if not isinstance(d, nn.Identity)]
        with torch.no_grad():
            m(torch.randn(4, 3, 32, 32))
        for h in hs:
            h.remove()
        for nm, frac in sorted(seen.items()):
            check(f"{arch}/{nm}: 음수 비율 {frac:.2f}", frac < 0.01)


def check_collisions():
    """이미 있는 산출물과 이름이 겹치면서 설정이 다른 것이 없는가."""
    print("\n[6] 산출물 무결성 — 이름 충돌과 에폭 혼입")
    hist = glob.glob("checkpoints/*_history.csv")
    ep = {}
    for f in hist:
        rid = os.path.basename(f)[:-len("_history.csv")]
        ep[rid] = sum(1 for _ in open(f)) - 1
    odd = {k: v for k, v in ep.items() if v not in (200,)}
    warn(f"200에폭이 아닌 런 {len(odd)}개 (집계 시 반드시 필터할 것)",
         len(odd) == 0, ", ".join(f"{k}({v})" for k, v in list(odd.items())[:4]) + " ...")
    # config.json 이 없는 구 산출물
    nocfg = [k for k in ep if not os.path.exists(f"checkpoints/{k}_config.json")]
    warn(f"config.json 없는 구 산출물 {len(nocfg)}개 (덮어쓰기 방지가 에폭 수로만 동작)",
         len(nocfg) == 0)
    check(f"체크포인트 {len(hist)}개 읽기 가능", len(hist) > 0)


def check_flags_raise():
    """지원하지 않는 플래그가 조용히 무시되지 않고 오류를 내는가."""
    print("\n[7] 유령 어블레이션 — 미지원 플래그가 오류를 내는가")
    from sdrop import build_sdrop
    from model import build_model
    from sdrop_vit import build_sdrop_vit
    cases = [
        ("sgridlc + schedule", lambda: build_sdrop('sgridlc', 0.3, schedule='linear')),
        ("resnet56 + width_mult", lambda: build_model('resnet56', 100, 'sdrop', 0.1, ['L4'], width_mult=0.5)),
        ("resnet56 + L1", lambda: build_model('resnet56', 100, 'sdrop', 0.1, ['L1'])),
        ("vit + L1", lambda: build_sdrop_vit(num_classes=100, method='sdrop_vit', drop_rate=0.3, layers=['L1'])),
        ("vit_full + layers", lambda: build_sdrop_vit(num_classes=100, method='sdrop_vit_full', drop_rate=0.3, layers=['L4'])),
        # WRN 은 pre-activation 이라 L2/L3 출력이 raw signed (음수 52%) 다.
        # post-ReLU 를 가정하는 SDrop 점수와 맞지 않아 같은 플래그가 다른 개입이 된다.
        ("wrn28_10 + L2", lambda: build_model('wrn28_10', 100, 'sdrop', 0.1, ['L2'])),
        ("wrn28_10 + L3", lambda: build_model('wrn28_10', 100, 'sdrop', 0.1, ['L3'])),
        # baseline 모듈은 SDrop 전용 플래그를 받지 않는데 run_id 는 태그를 붙인다.
        ("wcd + schedule", lambda: build_model('resnet18', 100, 'wcd', 0.1, ['L4'], schedule='linear')),
        ("dropblock + rescale", lambda: build_model('resnet18', 100, 'dropblock', 0.1, ['L4'], rescale=True)),
        ("senet + participation", lambda: build_model('resnet18', 100, 'senet', 0.1, ['L4'], participation=0.1)),
    ]
    for name, fn in cases:
        try:
            fn(); check(f"{name} → 오류", False, "조용히 통과했다")
        except (ValueError, SystemExit):
            check(f"{name} → 오류", True)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    torch.manual_seed(0)
    print("=" * 66)
    print("  실험 사전점검")
    print("=" * 66)
    check_budget()
    check_rescale()
    check_modules_bite()
    check_params_before_forward()
    if not quick:
        check_smoke()
    check_insertion_activation()
    check_collisions()
    check_flags_raise()
    print("\n" + "=" * 66)
    if FAIL:
        print(f"  실패 {len(FAIL)}건 — 실험을 걸지 말 것")
        for f in FAIL:
            print(f"    ✘ {f}")
        sys.exit(1)
    print(f"  전 항목 통과" + (f" (경고 {len(WARN)}건)" if WARN else ""))
