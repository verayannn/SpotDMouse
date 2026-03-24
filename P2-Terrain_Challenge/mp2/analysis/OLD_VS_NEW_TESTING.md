# How to test if your new code/policy is better than the old one

Use the same test conditions for both policies, then compare quantitative metrics.

## 0) One-command runner (recommended)

Use the new runner to generate both suites and score them in one shot:

```bash
python3 P2-Terrain_Challenge/mp2/analysis/run_ab_benchmark.py \
  --old-policy /path/to/old_policy.pt \
  --new-policy /path/to/new_policy.pt \
  --out-root /tmp/mp2_ab \
  --duration 10 \
  --settle 2.0
```

This creates:
- `/tmp/mp2_ab/old/openloop_scheduled_*.csv`
- `/tmp/mp2_ab/new/openloop_scheduled_*.csv`
and then runs `ab_test_policies.py` automatically.

## 1) Open-loop A/B test (fast, no robot)

Generate the same 7 command rollouts for each policy:

```bash
# OLD policy
python3 P2-Terrain_Challenge/mp2/misc/test_policy_openloop.py \
  --policy /path/to/old_policy.pt --compare --duration 10

# NEW policy
python3 P2-Terrain_Challenge/mp2/misc/test_policy_openloop.py \
  --policy /path/to/new_policy.pt --compare --duration 10
```

Move OLD outputs into one folder and NEW outputs into another (each must include all
`openloop_scheduled_*.csv` files).

Then run the A/B scorer:

```bash
python3 P2-Terrain_Challenge/mp2/analysis/ab_test_policies.py \
  --old /path/to/old_rollouts \
  --new /path/to/new_rollouts \
  --settle 2.0
```

### What these metrics mean

- **separability** (higher is better): does the policy produce distinct behavior for different commands?
- **limit_hit_rate** (lower is better): how often joints saturate at limits.
- **smoothness_jerk** (lower is better): how abrupt action changes are.
- **total_score** (higher is better): weighted combination of the above.

## 2) Real robot A/B test (required for final decision)

For each policy, record repeated runs (same battery level window, floor, command, and duration):

```bash
python3 P2-Terrain_Challenge/mp2/misc/collect_walk_forward.py \
  --policy /path/to/policy.pt --vx 0.15 --duration 10
```

Then compare each real run to open-loop proxy using:

```bash
python3 P2-Terrain_Challenge/mp2/misc/compare_real_vs_openloop.py \
  --real /path/to/real_run.csv --openloop /path/to/openloop_cmd_0.15_0.0_0.0.csv
```

## 3) Acceptance criteria (practical)

Treat NEW as better only if all are true:

1. Open-loop `total_score` improves consistently across at least 3 seeds/runs.
2. On real robot, no increase in falls/stumbles and no overheating/current-limit events.
3. Real-vs-openloop divergence does not worsen materially.
4. Command tracking quality (forward/lateral/yaw) is equal or better.

## 4) Common pitfalls

- Comparing runs with different `HW_SCALE`, `pd_delay_steps`, or control frequency.
- Different warmup/settle times when computing stats.
- Single-run conclusions (use repeated trials and report mean ± std).
