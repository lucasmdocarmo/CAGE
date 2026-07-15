# `scripts/ops/` — operator tooling

Unnumbered on purpose. The numbered stages (`1_setup` … `6_teardown`) are the happy path of a run;
these are tools that run **alongside** any stage — same rationale as `checks/` and `lib/`.

Grounded in the `gcp-background-tasks` skill. The rule it exists to enforce:

> **An agent's shell is not a terminal someone is watching.** A blocking command that can exceed the
> tool timeout will be cut off mid-flight, the agent will see a truncated failure, and it will retry —
> now two setups race on one GPU box. Meanwhile the original keeps billing.

## The contract

```
SUBMIT → write the handle (remote PID) + state file BEFORE reporting success
POLL   → short bounded checks, backoff, an ENFORCED hard deadline
STREAM → bounded: tail -n N / grep. Never firehose a log into the context window
FINISH → read the EXIT CODE, not "the command returned"
REAP   → kill the process, verify with a read-only sweep that nothing is left billing
```

## Files

| File | Use it for |
|---|---|
| `remote_job.sh` | Any long command **on the GPU VM over SSH** (setup, sweeps, stats). Detaches it, records the remote PID/log/exit-status, writes `.agent/tasks/<name>.remote.json` storing `poll_cmd`/`cancel_cmd` **verbatim** so a later turn (or a compacted context) can resume knowing only the job name. |
| `gpu_vm.sh` | `create` — L4 **zone-hunt + shape fallback** (capacity is scarce) with an `agent-run` label; `sweep` — prove we're at $0. |

**Local** background work (not on the VM): use the skill's helper, already in this repo at
`.claude/skills/gcp-background-tasks/scripts/bgtask.sh` — deliberately not copied here (a vendored
duplicate drifts; same failure mode as the old `companion_images/`).

## Typical run

```bash
# provision (labels + zone-hunt; writes .agent/cage_zone)
scripts/ops/gpu_vm.sh create cage-gpu            # tries g2-standard-8, falls back to -4

# long remote work — never blocks the turn
scripts/ops/remote_job.sh submit setup 'cd ~/CAGE && env HF_HUB_DOWNLOAD_TIMEOUT=30 bash scripts/1_setup/setup_gpu_cloud.sh' 3600
scripts/ops/remote_job.sh status setup           # RUNNING | DONE(0) | FAILED(n) | KILLED | CRASHED
scripts/ops/remote_job.sh tail   setup 40        # bounded
scripts/ops/remote_job.sh grep   setup           # error triage
scripts/ops/remote_job.sh wait   setup 3600      # backoff → HARD deadline
scripts/ops/remote_job.sh fetch  setup           # pull the log local for the record

# teardown: syncs -> collects logs -> verifies sentinel -> PULLS results to local results/
# -> only then deletes. Fail-closed on a missing sentinel OR an incomplete local pull.
bash scripts/6_teardown/teardown_vm.sh cage-gpu "$(cat .agent/cage_zone)"
scripts/ops/gpu_vm.sh sweep          # PROVE $0
```

## Non-negotiables (each one cost us real time)

- **`CLOUDSDK_CORE_DISABLE_PROMPTS=1`** — a confirmation prompt in a TTY-less background shell hangs
  forever and emits nothing. The #1 cause of a "stuck" task.
- **Kill by recorded PID, never `pkill -f <script>`** — that pattern matches the SSH command's own
  shell and kills the session (exit 255). Use the `[b]`-bracket trick if you must match by name.
- **Don't build an ssh command in a variable** (`SSH="gcloud …"; $SSH …`) — under zsh an unquoted var
  does **not** word-split, so every check silently returns empty and the poller spins forever.
  Same trap as `for Z in $ZONES`: use a literal list, or a bash script with a proper shebang.
- **Label every billable resource** (`--labels=agent-run=<id>`) — the label is the reaping key.
  `gpu_vm.sh sweep` proves $0; an unlabeled orphan GPU VM is the most expensive failure here.
- **Tar+scp, don't `scp --recurse`** many small files — the recursive form blew a 2-min timeout on
  ~40 snapshots. `git archive` → one tarball is faster and reproducible.
- **Enforce the deadline.** A task past its deadline gets cancelled, not "it's probably nearly done".
- **Pull results local BEFORE teardown, never after.** Teardown is irreversible; until the run exists
  in a third place (VM + GCS + **local**), it is one failed sync away from gone. Pulling afterwards
  only works if the mirror happened to be complete. `teardown_vm.sh` step `[4/6]` does this and fails
  closed; `CAGE_SKIP_LOCAL_PULL=1` opts out (and leaves you with no local copy).

## Shape note (matters for sweep planning)

`g2-standard-4` = 4 vCPU/16 GB · `g2-standard-8` = 8 vCPU/32 GB — **both have one L4 (24 GB)**.
CAGE's quality scoring (LettuceDetect/NLI/BERTScore/e5) runs **on CPU** while vLLM holds the GPU, so
vCPU count sets sweep wall-clock: the `-4` shape scores ~2× slower. Prefer `-8` for a real sweep;
`-4` is fine for validation. On 2026-07-15 only `-4` had capacity anywhere.

`.agent/` is runtime state (gitignored), not code.
