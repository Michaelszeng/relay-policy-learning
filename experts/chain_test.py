"""Exhaustive 4-subtask chain test. Reports per-combo failure rate and cause."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adept_envs"))
import numpy as np, gym, adept_envs, controller, expert

# fixed canonical order; each combo runs its members in this relative order
ORDER = ["microwave", "kettle", "bottomknob", "topknob", "light", "hinge", "slide"]
SEEDS = 5
PER = 200

def rollout(env, seq):
    env.reset()
    idx = 0; max_steps = PER * len(seq)
    stall_state = None; dwell = {}
    for _ in range(max_steps):
        if idx >= len(seq): break
        label = seq[idx]
        sub, fs = expert.compute_state(env, label)
        g, tp, tr = expert.compute_action(env, sub, fs)
        if fs == "return-to-home":
            a = controller.joint_pose_to_action(env, expert.RESET_ARM_QPOS, g)
        else:
            a = controller.pose_to_action(env, tp, tr, g)
        dwell[(idx, fs)] = dwell.get((idx, fs), 0) + 1
        env.step(a)
        if expert.is_done(env, label) and expert.at_home(env):
            idx += 1
    done = {s: bool(expert.is_done(env, s)) for s in seq}
    # diagnose the stall: the subtask the oracle was stuck on
    if idx < len(seq):
        stuck = seq[idx]
        _, fs = expert.compute_state(env, stuck)
        goal_ok = bool(expert.is_done(env, stuck))
        home_ok = bool(expert.at_home(env))
        # the state it spent the most steps in for this idx (the "stuck" phase)
        phase = max((k for k in dwell if k[0]==idx), key=lambda k: dwell[k], default=(idx,fs))[1]
        cause = (f"goal-not-achieved" if not goal_ok
                 else f"achieved-but-not-home")
        stall = dict(subtask=stuck, pos=idx, phase_now=fs, stuck_phase=phase,
                     goal_ok=goal_ok, home_ok=home_ok, cause=cause)
    else:
        stall = None
    return all(done.values()) and idx==len(seq), done, stall

def main():
    env = gym.make("kitchen_relax-v1").unwrapped
    subs = ORDER[:]  # all 7
    combos = list(itertools.combinations(subs, 4))
    # order each combo by canonical ORDER
    combos = [tuple(sorted(c, key=ORDER.index)) for c in combos]
    print(f"Testing {len(combos)} combos x {SEEDS} seeds, order={ORDER}\n", flush=True)
    results = []
    for ci, combo in enumerate(combos):
        fails = []
        for seed in range(SEEDS):
            np.random.seed(seed)
            ok, done, stall = rollout(env, list(combo))
            if not ok:
                fails.append((seed, done, stall))
        nf = len(fails)
        results.append((combo, nf, fails))
        tag = "OK" if nf==0 else f"FAIL {nf}/{SEEDS}"
        cause = ""
        if fails:
            s = fails[0][2]
            cause = f" | stall@{s['subtask']}(pos{s['pos']}):{s['stuck_phase']} [{s['cause']}]"
        print(f"[{ci+1:2d}/{len(combos)}] {'+'.join(combo):45s} {tag}{cause}", flush=True)
    env.close_env(); env.close()

    # ---- aggregate report ----
    print("\n\n========== SUMMARY ==========", flush=True)
    nfail_combos = sum(1 for _,nf,_ in results if nf>0)
    total_runs = len(results)*SEEDS
    total_fail = sum(nf for _,nf,_ in results)
    print(f"Combos: {len(results)}  | combos with >=1 failure: {nfail_combos}")
    print(f"Runs: {total_runs}  | failed runs: {total_fail} ({100*total_fail/total_runs:.1f}%)\n")

    # blame: which subtask is the stall point, and cause
    from collections import Counter
    blame = Counter(); cause_c = Counter(); blame_cause = Counter()
    for combo,nf,fails in results:
        for seed,done,stall in fails:
            if stall:
                blame[stall['subtask']] += 1
                cause_c[stall['cause']] += 1
                blame_cause[(stall['subtask'], stall['stuck_phase'], stall['cause'])] += 1
    print("Failures blamed on (stall subtask):")
    for s,c in blame.most_common(): print(f"  {s:12s}: {c}")
    print("\nFailure causes:")
    for s,c in cause_c.most_common(): print(f"  {s:22s}: {c}")
    print("\nDetailed (subtask, stuck_phase, cause): count")
    for k,c in blame_cause.most_common(): print(f"  {str(k):60s}: {c}")

    print("\nFailing combos (rate):")
    for combo,nf,fails in results:
        if nf>0:
            s=fails[0][2]
            print(f"  {'+'.join(combo):45s} {nf}/{SEEDS}  stall@{s['subtask']}:{s['stuck_phase']} [{s['cause']}]")

if __name__ == "__main__":
    main()
