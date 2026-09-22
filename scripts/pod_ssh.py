"""Print '<host> <port>' for a pod's direct SSH (port 22/tcp), from runpodctl JSON.

    python scripts/pod_ssh.py <POD_ID>

Tries `runpodctl ssh info` first, then falls back to `runpodctl pod get` and looks for the
port mapping whose private port is 22. Tolerant of key naming differences between versions.
"""
import json
import os
import subprocess
import sys

RP = os.environ.get("RUNPODCTL", os.path.join(os.path.dirname(__file__), "..", "tools", "runpodctl.exe"))


def run(*args):
    out = subprocess.run([RP, *args, "-o", "json"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def find(obj):
    for d in walk(obj):
        # port-mapping shape: {"ip": ..., "privatePort": 22, "publicPort": 12345, "type": "tcp"}
        if d.get("privatePort") == 22 and d.get("publicPort") and (d.get("ip") or d.get("publicIp")):
            return d.get("ip") or d.get("publicIp"), d["publicPort"]
    for d in walk(obj):
        host = d.get("host") or d.get("ip") or d.get("publicIp")
        port = d.get("port") or d.get("sshPort") or d.get("publicPort")
        if host and port and "proxy" not in str(host):
            return host, port
    return None


def main():
    pod_id = sys.argv[1]
    for args in (("ssh", "info", pod_id), ("pod", "get", pod_id)):
        data = run(*args)
        if data:
            hit = find(data)
            if hit:
                print(hit[0], hit[1])
                return 0
    print("could not find a 22/tcp mapping; was the pod created with --ports '8000/http,22/tcp'?", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
