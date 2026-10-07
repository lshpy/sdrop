#!/usr/bin/env python3
"""RunPod 네트워크 볼륨(sdrop-results)에서 결과 CSV를 로컬로 동기화한다.

    python3 scripts/sync_volume.py            # 새 파일만 받기
    python3 scripts/sync_volume.py --all      # 크기가 다르면 덮어쓰기까지

파드를 켜지 않아도 된다 — S3 호환 API로 볼륨에 직접 붙기 때문에 GPU 과금이 없다.
자격증명은 ~/.runpod_s3_env 에 있다(600 권한, 저장소에 넣지 말 것).
"""
import os, sys, boto3, botocore, pathlib, urllib.parse

ENVF = pathlib.Path.home() / ".runpod_s3_env"
for line in ENVF.read_text().splitlines():
    if line.startswith("export "):
        k, v = line[7:].split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"'))

DEST = pathlib.Path(__file__).resolve().parent.parent / "paper_results" / "raw_pod"
PREFIX = "sdrop/checkpoints/"
overwrite = "--all" in sys.argv

s3 = boto3.client(
    "s3", endpoint_url=os.environ["RUNPOD_S3_ENDPOINT"],
    aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"],
    region_name="EUR-IS-1",
    config=botocore.config.Config(signature_version="s3v4"),
)
DEST.mkdir(parents=True, exist_ok=True)

new = skipped = 0
failed = []
# EncodingType="url" 을 주지 않으면 키 안의 '+'(예: L3+L4)가 공백으로 디코딩되어
# 되돌아오고, 그 키로는 get_object 가 NoSuchKey 를 낸다. 직접 unquote 한다.
for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=os.environ["RUNPOD_S3_BUCKET"], Prefix=PREFIX, EncodingType="url"):
    for obj in page.get("Contents", []):
        obj = dict(obj, Key=urllib.parse.unquote(obj["Key"]))
        if not obj["Key"].endswith("_history.csv"):
            continue
        out = DEST / obj["Key"].rsplit("/", 1)[-1]
        if out.exists() and (not overwrite or out.stat().st_size == obj["Size"]):
            skipped += 1
            continue
        # download_file 은 내부적으로 head_object 를 부르는데, RunPod S3 는 키에 '+'가
        # 들어가면(예: L3+L4) 여기서 404 를 준다. get_object 로 직접 스트리밍한다.
        try:
            body = s3.get_object(Bucket=os.environ["RUNPOD_S3_BUCKET"], Key=obj["Key"])["Body"]
            out.write_bytes(body.read())
            new += 1
        except Exception as e:
            failed.append((obj["Key"], type(e).__name__))

print(f"받음 {new}개, 건너뜀 {skipped}개  →  {DEST}")
print(f"로컬 총 CSV: {len(list(DEST.glob('*_history.csv')))}개")
if failed:
    print(f"실패 {len(failed)}개:")
    for k, e in failed[:10]:
        print("  ", k, e)
