#!/usr/bin/env python3
"""RunPod 파드 생성 (REST v1). 네트워크 볼륨 sdrop-results 를 /workspace 에 붙이고 entry 스크립트를 실행한다.
   python3 scripts/create_pod.py <name> <entry_script.sh> [MAX_HOURS]
파드는 entry 스크립트가 끝나면 스스로 terminate 한다 (RUNPOD_API_KEY 를 env 로 넘김)."""
import json, os, pathlib, sys, urllib.request
env = dict(l.replace("export ","").split("=",1) for l in (pathlib.Path.home()/".runpod_env").read_text().split("\n") if "=" in l)
KEY = env["RUNPOD_API_KEY"].strip().strip('"')
name, entry = sys.argv[1], sys.argv[2]; max_hours = sys.argv[3] if len(sys.argv) > 3 else "4"
body = {
  "name": name, "cloudType": "SECURE", "computeType": "GPU",
  "gpuTypeIds": os.environ.get("GPU_TYPES","NVIDIA GeForce RTX 5090").split(";"), "gpuCount": 1,
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "containerDiskInGb": 40, "volumeInGb": 0,
  "networkVolumeId": "evlgsb1vno", "volumeMountPath": "/workspace",
  "dataCenterIds": ["EUR-IS-1"], "ports": ["22/tcp"],
  "env": {"RUNPOD_API_KEY": KEY, "MAX_HOURS": max_hours,
          **dict(kv.split("=", 1) for kv in os.environ.get("POD_ENV", "").split(",") if "=" in kv)},  # e.g. POD_ENV="SHARD=0,NSHARD=4"
  "dockerStartCmd": ["bash", "-c", f"cd /workspace/sdrop && bash {entry}; sleep 60"],
}
import subprocess
out = subprocess.run(["curl","-s","-A","sdrop-cli/1.0","-X","POST","https://rest.runpod.io/v1/pods","-H","Content-Type: application/json",
                      "-H",f"Authorization: Bearer {KEY}","-d",json.dumps(body)],capture_output=True,text=True).stdout
try: r = json.loads(out)
except Exception: print("raw:", out[:500]); sys.exit(1)
if "id" not in r: print("error:", r); sys.exit(1)
print(json.dumps({k: r.get(k) for k in ("id","name","desiredStatus","costPerHr","machine")}, ensure_ascii=False))
(pathlib.Path.home()/".runpod_last_pod").write_text(r["id"])
