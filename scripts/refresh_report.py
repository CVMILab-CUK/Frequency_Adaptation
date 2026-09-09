#!/usr/bin/env python3
"""Regenerate the report's data block from whatever results exist on disk.

Rewrites only the `const D = {...}` line inside the published page, so the
report can be re-published unchanged as new runs finish.
"""
import glob, json, os, re, sys

ROOT = "/home/work/model/Frequency_Adaptation"
PAGE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FA_REPORT_PAGE")

def table(name):
    p = os.path.join(ROOT, "results", name, "results.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))["per_cutoff"]
    return [{"r": float(k), "SC": round(v["SC"]["mean"],4), "sem": round(v["SC"]["sem"],4),
             "EdgeF1": round(v["EdgeF1"]["mean"],4), "LPIPS": round(v["LPIPS"]["mean"],4),
             "CLIP": round(v["CLIP"]["mean"],2), "Div": round(v["Diversity"]["mean"],4),
             "FID": round(v["FID"],2) if v["FID"] is not None else None}
            for k, v in sorted(d.items(), key=lambda kv: float(kv[0]))]

def curve(name, cuts=("0.0","0.1","0.3","0.7")):
    p = os.path.join(ROOT, "log_dirs", f"{name}_valcurve.jsonl")
    if not os.path.exists(p): return None
    rows = []
    for line in open(p):
        rec = json.loads(line); e = {"step": rec["step"]}; ok = False
        for c in cuts:
            k = f"valid/sc_at_r{c}"
            if k in rec: e[c] = round(rec[k], 4); ok = True
        if ok: rows.append(e)
    return rows

new = {
  "e1": table("E1"), "e3h": table("E3h"),
  "e1off": table("E1_adapter_off"), "e3hoff": table("E3h_adapter_off"),
  "e9b": table("E9b"), "e9a": table("E9a"),
  "e9boff": table("E9b_adapter_off"), "e9aoff": table("E9a_adapter_off"),
  "curveE1": curve("FA_SD15_r0-1"), "curveE3h": curve("E3h_zero_init_gate"),
  "curveE9b": curve("E9b_source256"), "curveE9a": curve("E9a_conv_encoder"),
}
old = json.loads(re.search(r"const D = (\{.*?\});", open(PAGE).read(), re.S).group(1))
for k in ("vae", "energy"):
    new[k] = old[k]
for k, v in list(new.items()):
    if v is None: new[k] = old.get(k)

s = open(PAGE).read()
s = re.sub(r"const D = \{.*?\};", "const D = " + json.dumps(new) + ";", s, flags=re.S)
open(PAGE, "w").write(s)
done = [k for k in ("e1","e3h","e9b","e9a") if new.get(k)]
print(f"report data refreshed; runs present: {done}")
