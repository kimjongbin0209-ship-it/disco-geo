"""Summary numbers and README figures for the P1 and P1d stages.

Usage (repository root, after copying both result archives back):
    python scripts/plot_results.py --pilot runs/pilot_results/runs \
        --diagnostics runs/diagnostics_results/runs/diagnostics \
        --diagnostics-summary runs/diagnostics_results/runs/evaluation/diagnostics/diagnostics.json

Only training logs are read. Every training step is a sampled action of the current
policy, so the reward rate over a span of training is an on-policy estimate (rewards per
1000 environment steps on Memory Maze, per trial on the cue controls). P1d's per-run
numbers (reward, credit gap, entropy) come from diagnostics.json written by
scripts/diagnose_stage.py; the curves come from each run's metrics.jsonl.

The uniform-policy baseline is the pooled last quarter of the AC+aux entropy-0.01 runs,
whose policy stayed uniform (entropy 1.78 of ln 6 = 1.79, every action at 0.17). The
scripted references in diagnostics.json used only 32 episodes (the uniform one earned 5
rewards), too few for a rate-ratio CI. Writes <output>/summary.json and light/dark PNGs
in <output>/figures.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SEEDS = (0, 1, 2)
UNIFORM_ENTROPY = math.log(6)
METHODS = {"actor_critic_aux": "AC+aux", "disco": "Disco103"}
# Reference palette slots 1 (blue) and 2 (orange), each stepped for its surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series={"actor_critic_aux": "#2a78d6", "disco": "#eb6834"}),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series={"actor_critic_aux": "#3987e5", "disco": "#d95926"}),
}
MAZE_CONDITIONS = [  # (label, method, condition)
    ("AC+aux\nentropy 0.01", "actor_critic_aux", "diag_fresh"),
    ("AC+aux\nentropy 1e-3", "actor_critic_aux", "diag_entropy_1e-3"),
    ("AC+aux\nentropy 0", "actor_critic_aux", "diag_entropy_0"),
    ("Disco103\nlr 3e-4", "disco", "diag_fresh"),
    ("Disco103\nlr 3e-3", "disco", "diag_lr_3e-3"),
    ("Disco103\nlr 1e-2", "disco", "diag_lr_1e-2"),
]
MAZE_X = [0, 1, 2, 3.5, 4.5, 5.5]
CONTROL_TASKS = [("control_bandit", "One-answer bandit"), ("control_reactive", "Reactive cue"),
                 ("control_memory", "Cue held for 8 steps")]
CONTROL_LEARNERS = [("AC+aux\nlr 3e-4", "actor_critic_aux", ""), ("Disco103\nlr 3e-4", "disco", ""),
                    ("Disco103\nlr 1e-2", "disco", "_lr_1e-2")]


def rows_of(run):
    with (run / "metrics.jsonl").open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def name_parts(name):
    method, rest = name.split("-", 1)
    condition, seed = rest.rsplit("-seed", 1)
    return method, condition, int(seed)


def binned(rows, size):
    """Per bin of `size` updates: end transitions (millions), mean entropy, rewards per 1000 steps."""
    out = []
    for start in range(0, len(rows) - size + 1, size):
        part = rows[start:start + size]
        steps = part[-1]["transitions"] - (rows[start - 1]["transitions"] if start else 0)
        events = sum(row["reward_events_this_chunk"] for row in part)
        out.append((part[-1]["transitions"] / 1e6, np.mean([row["policy_entropy"] for row in part]),
                    1000 * events / steps))
    return np.array(out)


def pilot_summary(directory):
    runs = {}
    for run in sorted(p for p in directory.iterdir() if (p / "metrics.jsonl").exists()):
        rows = rows_of(run)
        start = 3 * len(rows) // 4
        tail = rows[start:]
        events = sum(row["reward_events_this_chunk"] for row in tail)
        steps = tail[-1]["transitions"] - rows[start - 1]["transitions"]
        method, condition, seed = name_parts(run.name)
        runs[run.name] = dict(method=method, condition=condition, seed=seed,
                              transitions=rows[-1]["transitions"],
                              last_quarter_reward_events=events, last_quarter_steps=steps,
                              last_quarter_reward_per_1000=1000 * events / steps,
                              entropy_min=min(row["policy_entropy"] for row in rows),
                              entropy_last_quarter=float(np.mean([row["policy_entropy"] for row in tail])))
    pooled = {}
    for method in METHODS:
        group = [r for r in runs.values() if r["method"] == method]
        events = sum(r["last_quarter_reward_events"] for r in group)
        steps = sum(r["last_quarter_steps"] for r in group)
        pooled[method] = dict(runs=len(group), last_quarter_reward_per_1000=1000 * events / steps,
                              entropy_last_quarter=[round(r["entropy_last_quarter"], 3) for r in group])
    return dict(runs=runs, pooled=pooled)


def diagnostics_summary(path):
    data = json.loads(path.read_text())
    runs = {r["run"]: r for r in data["runs"]}

    def group(method, condition):
        return [runs[f"{method}-{condition}-seed{s}"] for s in SEEDS]

    baseline = group("actor_critic_aux", "diag_fresh")
    base_events = sum(r["reward"]["last_quarter_events"] for r in baseline)
    base_steps = sum(r["reward"]["last_quarter_steps"] for r in baseline)
    maze = {}
    for label, method, condition in MAZE_CONDITIONS:
        rs = group(method, condition)
        events = sum(r["reward"]["last_quarter_events"] for r in rs)
        steps = sum(r["reward"]["last_quarter_steps"] for r in rs)
        entry = dict(label=label.replace("\n", " "), method=method, condition=condition,
                     learning_rate=rs[0]["learning_rate"], entropy_cost=rs[0]["entropy_cost"],
                     reward_per_1000_per_seed=[1000 * r["reward"]["last_quarter_events"]
                                               / r["reward"]["last_quarter_steps"] for r in rs],
                     pooled_reward_events=events, pooled_reward_per_1000=1000 * events / steps,
                     credit_gap_per_seed=[r["credit"]["update_logp_gain"]["gap"] for r in rs],
                     credit_gap_se_per_seed=[r["credit"]["update_logp_gain"]["gap_se"] for r in rs],
                     entropy_min_per_seed=[r["entropy"]["min"] for r in rs],
                     entropy_last_quarter_per_seed=[r["entropy"]["last_quarter"] for r in rs])
        if rs[0] is not baseline[0]:
            ratio = (events / steps) / (base_events / base_steps)
            half = 1.96 * math.sqrt(1 / events + 1 / base_events)
            entry["ratio_vs_uniform_policy"] = ratio
            entry["ratio_ci95"] = [ratio * math.exp(-half), ratio * math.exp(half)]
        maze[f"{method}-{condition}"] = entry
    controls = {}
    for task, title in CONTROL_TASKS:
        for label, method, suffix in CONTROL_LEARNERS:
            rs = group(method, task + suffix)
            values = [r["reward"]["per_trial_last_quarter"] for r in rs]
            controls[f"{method}-{task}{suffix}"] = dict(
                task=title, learner=label.replace("\n", " "), method=method,
                reward_per_trial_per_seed=values, criterion=rs[0]["reward"]["criterion"],
                chance=rs[0]["reward"]["chance"], passed=sum(r["reward"]["passed"] for r in rs),
                entropy_last_quarter_per_seed=[r["entropy"]["last_quarter"] for r in rs],
                credit_gap_per_seed=[r["credit"]["update_logp_gain"]["gap"] for r in rs])
    references = {name: dict(policy=r["policy"], episodes=r["episodes"], rewards=r["total_reward"],
                             reward_per_1000=r["reward_per_1000_steps"])
                  for name, r in data["references"].items()}
    uniform = dict(source="AC+aux entropy 0.01, pooled last quarter (policy stayed uniform)",
                   reward_events=base_events, steps=base_steps,
                   reward_per_1000=1000 * base_events / base_steps)
    return dict(uniform_policy_baseline=uniform, maze=maze, controls=controls,
                scripted_references=references)


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": t["ink"], "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "axes.edgecolor": t["axis"], "axes.labelcolor": t["secondary"],
        "text.color": t["ink"], "xtick.color": t["axis"], "ytick.color": t["axis"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["grid"], "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "ytick.major.size": 0, "legend.frameon": False, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })
    return t


def reference_line(ax, t, y, text=None, where="left"):
    """A muted horizontal reference, labelled inside ("left"), in the right margin ("outside"),
    or centred on a data x where no mark sits."""
    ax.axhline(y, color=t["muted"], linewidth=0.9, zorder=1)
    if text is None:
        return
    if where == "left":
        x, align, va, transform = 0.005, "left", "bottom", ax.get_yaxis_transform()
    elif where == "outside":
        x, align, va, transform = 1.01, "left", "center", ax.get_yaxis_transform()
    else:
        x, align, va, transform = where, "center", "bottom", ax.transData
    ax.text(x, y, text, transform=transform, ha=align, va=va, fontsize=7.5, color=t["secondary"])


def method_legend(fig, t, y=1.0):
    handles = [Line2D([], [], color=t["series"][m], marker="o", markersize=6, linewidth=1.6, label=label)
               for m, label in METHODS.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, y), ncol=2, fontsize=8.5,
               labelcolor=t["secondary"], handlelength=1.6)


def dot(ax, t, x, y, method, **extra):
    ax.plot([x], [y], "o", markersize=6.5, color=t["series"][method], markeredgecolor=t["surface"],
            markeredgewidth=1.2, zorder=3, **extra)


def figure_pilot(t, directory, uniform_rate):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for run in sorted(p for p in directory.iterdir() if (p / "metrics.jsonl").exists()):
        method = name_parts(run.name)[0]
        rows = rows_of(run)
        entropy, reward = binned(rows, 100), binned(rows, 250)
        top.plot(entropy[:, 0], entropy[:, 1], color=t["series"][method], linewidth=1.2, alpha=0.85)
        bottom.plot(reward[:, 0], reward[:, 2], color=t["series"][method], linewidth=1.2, alpha=0.85)
    reference_line(top, t, UNIFORM_ENTROPY, "uniform\n(ln 6)", where="outside")
    reference_line(bottom, t, uniform_rate, f"uniform\npolicy\n{uniform_rate:.2f}", where="outside")
    top.set_ylabel("policy entropy (nats)")
    bottom.set_ylabel("reward per 1000 steps")
    bottom.set_xlabel("environment steps (millions)")
    bottom.set_ylim(bottom=0)
    top.set_title("P1: the policies drift and return to uniform; reward stays at the uniform rate")
    method_legend(fig, t)
    fig.text(0.0, -0.02, "12 runs, 5M steps: 6 per learner (fresh and teacher_only, 3 seeds each; teacher_only "
             "never distils, so its student trains like fresh). Bins of 100 (entropy) and 250 (reward) updates.",
             fontsize=7.5, color=t["muted"])
    fig.tight_layout()
    return fig


def figure_controls(t, summary):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.3), sharey=True)
    for ax, (task, title) in zip(axes, CONTROL_TASKS):
        for x, (label, method, suffix) in enumerate(CONTROL_LEARNERS):
            entry = summary["controls"][f"{method}-{task}{suffix}"]
            for offset, value in zip((-0.13, 0, 0.13), entry["reward_per_trial_per_seed"]):
                dot(ax, t, x + offset, value, method)
            ax.text(x, 1.06, f"pass {entry['passed']}/3", ha="center", fontsize=7.5, color=t["secondary"])
        # Label each line once, on the left where no dot sits; panel 2 repeats panel 1's lines.
        first, last = ax is axes[0], ax is axes[-1]
        reference_line(ax, t, 1 / 6, "chance" if first else None)
        reference_line(ax, t, entry["criterion"], f"pass line {entry['criterion']:g}" if first or last else None)
        ax.set_xticks(range(3), [label for label, _, _ in CONTROL_LEARNERS], fontsize=7.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(title)
    axes[0].set_ylabel("reward per trial, last quarter")
    method_legend(fig, t, y=1.06)
    fig.text(0.0, -0.04, "Synthetic cue-recall controls, 3 seeds each (dots). Bandit: 300k steps; reactive and "
             "memory: 2M steps.", fontsize=7.5, color=t["muted"])
    fig.tight_layout()
    return fig


def figure_maze(t, summary):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for x, (label, method, condition) in zip(MAZE_X, MAZE_CONDITIONS):
        entry = summary["maze"][f"{method}-{condition}"]
        for offset, value in zip((-0.14, 0, 0.14), entry["reward_per_1000_per_seed"]):
            dot(left, t, x + offset, value, method)
        left.plot([x - 0.3, x + 0.3], [entry["pooled_reward_per_1000"]] * 2, color=t["ink"], linewidth=1.4,
                  zorder=2)
        for offset, gap, se in zip((-0.14, 0, 0.14), entry["credit_gap_per_seed"], entry["credit_gap_se_per_seed"]):
            right.errorbar([x + offset], [1000 * gap], yerr=[1000 * se], fmt="none", ecolor=t["series"][method],
                           elinewidth=1.2, capsize=0, zorder=2)
            dot(right, t, x + offset, 1000 * gap, method)
    uniform = summary["uniform_policy_baseline"]["reward_per_1000"]
    forward_left = summary["scripted_references"]["forward_left_0.5"]
    # The gap between the two learners' groups is free of dots.
    reference_line(left, t, uniform, f"uniform\npolicy {uniform:.2f}", where=(MAZE_X[2] + MAZE_X[3]) / 2)
    reference_line(left, t, forward_left["reward_per_1000"],
                   f"scripted forward-left bias ({forward_left['rewards']:g} rewards, rough)")
    right.axhline(0, color=t["axis"], linewidth=0.9, zorder=1)
    for ax in (left, right):
        ax.set_xticks(MAZE_X, [label for label, _, _ in MAZE_CONDITIONS], fontsize=7.5)
        ax.set_xlim(-0.6, 6.1)
    left.set_ylim(0, 0.48)
    left.set_ylabel("reward per 1000 steps, last quarter")
    right.set_ylabel("credit gap (10⁻³ nats per update)")
    left.set_title("Reward rate (dots: seeds; bar: pooled)")
    right.set_title("Credit gap ± block SE")
    method_legend(fig, t, y=1.06)
    fig.text(0.0, -0.04, "Memory Maze 9x9, P1d, 2M steps, 3 seeds. Credit gap: one update's gain in log pi(a_t) "
             "on steps followed by a reward within 8 steps, minus the gain on other steps.",
             fontsize=7.5, color=t["muted"])
    fig.tight_layout()
    return fig


def figure_maze_entropy(t, directory):
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.4), sharex=True, sharey=True)
    for ax, (label, method, condition) in zip(axes.flat, MAZE_CONDITIONS):
        for seed in SEEDS:
            curve = binned(rows_of(directory / f"{method}-{condition}-seed{seed}"), 20)
            ax.plot(curve[:, 0], curve[:, 1], color=t["series"][method], linewidth=1.2, alpha=0.85)
        ax.axhline(UNIFORM_ENTROPY, color=t["muted"], linewidth=0.9, zorder=1)
        ax.set_title(label.replace("\n", ", "), fontsize=8.5)
        ax.set_ylim(0, 1.95)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("policy entropy (nats)")
    axes[0, 0].text(0.02, UNIFORM_ENTROPY, "uniform (ln 6)", transform=axes[0, 0].get_yaxis_transform(),
                    va="bottom", fontsize=7.5, color=t["secondary"])
    method_legend(fig, t, y=1.05)
    fig.text(0.0, -0.03, "Memory Maze 9x9, P1d, 3 seeds per panel, bins of 20 updates. P1 used AC+aux entropy 0.01 "
             "and Disco103 lr 3e-4.", fontsize=7.5, color=t["muted"])
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("runs/pilot_results/runs"))
    parser.add_argument("--diagnostics", type=Path, default=Path("runs/diagnostics_results/runs/diagnostics"))
    parser.add_argument("--diagnostics-summary", type=Path,
                        default=Path("runs/diagnostics_results/runs/evaluation/diagnostics/diagnostics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/results"))
    options = parser.parse_args()
    summary = dict(note="Pilot and diagnostic evidence; not a study result.",
                   P1_learning_gate=pilot_summary(options.pilot),
                   P1d_diagnostics=diagnostics_summary(options.diagnostics_summary))
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures = options.output / "figures"
    figures.mkdir(exist_ok=True)
    diagnostics = summary["P1d_diagnostics"]
    uniform = diagnostics["uniform_policy_baseline"]["reward_per_1000"]
    for theme in THEMES:
        t = style(theme)
        suffix = "" if theme == "light" else "-dark"
        for name, fig in (("p1_learning_gate", figure_pilot(t, options.pilot, uniform)),
                          ("p1d_controls", figure_controls(t, diagnostics)),
                          ("p1d_maze_reward_credit", figure_maze(t, diagnostics)),
                          ("p1d_maze_entropy", figure_maze_entropy(t, options.diagnostics))):
            fig.savefig(figures / f"{name}{suffix}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {options.output / 'summary.json'} and {len(THEMES) * 4} figures in {figures}")


if __name__ == "__main__":
    main()
