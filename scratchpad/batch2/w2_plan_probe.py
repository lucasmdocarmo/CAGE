import json, subprocess, sys
from pathlib import Path
out = Path("scratchpad/batch2")
rows = []
for r in (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25):
    rows.append({"r": r, "demand_bytes": 10_000_000_000, "budget_bytes": int(r * 10_000_000_000),
                 "lambda_kv_rps": 2.0 * r, "lambda_compute_rps": None,
                 "lambda_star_pred_rps": 2.0 * r, "lambda_star_basis": "probe"})
ft = out / "w2_floor_probe.json"
ft.write_text(json.dumps({"schema": "floor-table-v1",
    "generated_inputs": {"model": "qwen3-14b", "engine": "vllm", "kv_dtype": "bf16", "grid": "anchor-fine"},
    "rows": rows}))
plan_path = out / "w2_plan_probe.json"
rc = subprocess.run([sys.executable, "scripts/3_run/run_campaign.py", "plan", "--session", "a",
                     "--floor-table", str(ft), "--window-duration-s", "300", "--out", str(plan_path)],
                    capture_output=True, text=True)
print("plan rc:", rc.returncode, rc.stdout.strip(), rc.stderr.strip()[-300:])
plan = json.loads(plan_path.read_text())
steps = plan["steps"]
sg_relaunch = next(s for s in steps if s["kind"] == "relaunch" and s["engine"] == "sglang")
sg_cell = next(s for s in steps if s["kind"] == "cell" and s["cellspec"]["engine"] == "sglang")
vl_cell = next(s for s in steps if s["kind"] == "cell" and s["cellspec"]["engine"] == "vllm")
hf_cell = next(s for s in steps if s["kind"] == "cell" and s["cellspec"]["engine"] == "hf")
def api(s): 
    a = s["argv"]; return a[a.index("--api-base") + 1] if "--api-base" in a else None
print("sglang relaunch env:", json.dumps(sg_relaunch["env"]), "api_base:", sg_relaunch["api_base"])
print("sglang cell --api-base:", api(sg_cell), "| row:", sg_cell["row_key"])
print("vllm   cell --api-base:", api(vl_cell), "| row:", vl_cell["row_key"])
print("hf     cell --api-base:", api(hf_cell), "| row:", hf_cell["row_key"])
print("header serving_shapes:", json.dumps({k: v for k, v in plan["serving_shapes"].items() if "port" in k or "api_base" in k}))
print("counts:", json.dumps(plan["counts"]["cells"]), plan["counts"]["windows"], plan["counts"]["relaunches"], plan["counts"]["blocked"])
# the load side accepts it, and a hand-pointed sglang cell is refused
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("run_campaign", "scripts/3_run/run_campaign.py")
m = importlib.util.module_from_spec(spec); sys.modules["run_campaign"] = m; spec.loader.exec_module(m)
print("load_plan cells:", m.load_plan(plan_path)["counts"]["cells"])
bad = json.loads(plan_path.read_text())
c = next(s for s in bad["steps"] if s["kind"] == "cell" and s["cellspec"]["engine"] == "sglang")
c["argv"][c["argv"].index("--api-base") + 1] = "http://localhost:8000"
bad_path = out / "w2_plan_probe_bad.json"; bad_path.write_text(json.dumps(bad))
try:
    m.load_plan(bad_path); print("REFUSAL MISSING")
except m.RunError as e:
    print("refused:", str(e).splitlines()[1][:220])
