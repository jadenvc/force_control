# Sanding Teleop: Contact-Jitter Investigation & Fixes

Summary of the changes made while chasing down force oscillation / haptic "jitter"
during sliding contact in the sanding teleop task (`sanding_teleop.py`,
`teleop_sanding.py`). Two independent problems were found and fixed: one on the
**sim contact side**, one on the **haptic device side**.

## 1. Sim-side: `--pad-softness` was silently inert (priority bug)

MuJoCo contact parameters (`solref`/`solimp`/`friction`) are **not combined**
between two contacting geoms — they're taken entirely from whichever geom has the
higher `priority`. (This is different from, say, friction coefficients in some
other engines which use a geometric-mean or max combination rule; MuJoCo's
`priority` mechanism is a deliberate escape hatch for exactly this situation —
letting one designated geom's material properties "win" a contact outright,
rather than getting diluted by averaging with whatever it touches.)
`_configure_pad_contact()` was only writing the softness interpolation onto the
pad geom (`priority=8`); the panel surface geom has `priority=10` (a "compiled
default" set when the panel XML was authored, unrelated to and predating the
softness feature) and was winning every contact. So every value of
`--pad-softness`, from `0.0` to `1.0`, was compiling into a `SandingEnv` that
used the *panel's* original, unmodified, unrelated `solref`/`solimp`/`friction`
values for every single contact — the knob had literally zero causal effect on
simulated behavior, silently, for as long as it existed. This was only
discovered by directly reading back `model.geom_solref[pad_geom_id]` vs.
measuring actual force response at different `--pad-softness` settings and
finding them identical.

**Fix:** `_configure_pad_contact()` now writes `solref`/`solimp`/`friction` to
`panel_surface_geom_id` as well (the geom that actually wins), in addition to
`pad_geom_id`.

## 2. Sim-side: contact was still too stiff even at max softness

Once the knob actually worked, the softest settings still weren't soft enough to
tame oscillation during fast lateral sliding. Iterated on `SOFT_PAD_SOLREF`:

| value | result |
|---|---|
| `(0.025, 2.0)` (original) | frequent oscillation/dropout during sliding |
| `(0.150, 3.0)` | oscillation resolved, but steady-state stiffness collapsed ~25x (same 1.1mm penetration that produced 18N now settled at 0.7N) — impractical, breaks force calibration |
| **`(0.060, 2.0)`** (chosen) | ~2.6x stiffness reduction (manageable), oscillation/dropout cut dramatically |

Measured on the real default code path (`tool_kp=16000`, `arm_damping=2.5`,
default properties) against the standard sliding-sweep test:
- **Dropouts: 944 → 0** (out of 3499 steps)
- **Force std: 29.5N → 16.4N** (~44% reduction)

`SOFT_PAD_SOLIMP_WIDTH` was left at `0.004` (close to compiled default) — widening
it to `0.010` broke small-penetration force calibration (dose stayed at 0.0 in
several tests, because normal ~1mm operating penetrations fell entirely inside the
new, much wider impedance ramp-up zone).

Default `pad_softness` raised `0.6 → 1.0` since the range is now actually useful.

**Why the fix works, mechanistically:** MuJoCo's `solref = (time_constant,
damping_ratio)` parameterizes the contact as an implicit second-order
constraint-force filter, roughly analogous to a critically/under/over-damped
spring-damper with natural frequency `ω_n ≈ 1/time_constant` (for
`damping_ratio=1`) reacting to constraint violation (penetration). Lengthening
`time_constant` (0.025s → 0.060s) lowers that effective bandwidth — the
constraint force responds more slowly to a given penetration change. During a
fast sliding stroke, the *rate* of penetration change (driven by lateral
velocity crossing surface micro-features / the pad's curvature) can momentarily
exceed what a fast/stiff contact can track smoothly, producing overshoot →
correction → overshoot (this is the oscillation/dropout signature). A slower
contact simply can't react fast enough to build up that overshoot in the first
place — but the same mechanism that filters out the *fast* transient also
attenuates the force response to a *sustained* penetration, which is why
steady-state stiffness dropped (a `time_constant` this long isn't "instant
spring, slow to fully settle" — the reduced bandwidth is real and applies at
DC too, not just at high frequency). That's the origin of the 25x stiffness
collapse at the too-soft `(0.150, 3.0)` setting, and why `(0.060, 2.0)` is a
genuine compromise rather than a free lunch: some of the desired steady-state
force sensitivity is deliberately being traded for sliding stability.

## 3. Test suite fixes (consequence of #1/#2)

Three tests assumed rigid contact (`penetration = force_target_n / tool_kp`), which
is no longer valid once the contact is deliberately much softer. Replaced with a
fixed, empirically-checked `4mm` penetration, and increased settle/search step
budgets to account for the contact's longer real settling time. All 13 tests pass.

## 4. Rejected approaches (tried, measured, reverted)

- **Operational-space (inertia-decoupled) task-space control.** The plain
  Jacobian-transpose controller commands `τ = Jᵀ·(Kp·e − Kd·ė)` — it maps a
  desired Cartesian force/wrench to joint torque, but it implicitly assumes the
  arm's effective inertia *as seen at the tool* is roughly uniform/isotropic. It
  isn't: `Λ(q) = (J(q)·M(q)⁻¹·J(q)ᵀ)⁻¹`, the operational-space inertia matrix,
  varies with configuration and is generally anisotropic (stiffer along some
  Cartesian directions than others depending on joint arrangement). The textbook
  fix (Khatib) is to decouple it: command `F = Λ(q)·(Kp·e − Kd·ė)` instead, so
  the same gains produce the same *acceleration* response regardless of arm
  pose/direction.

  This was implemented properly — computed `M(q)` via `mujoco.mj_fullM`,
  inverted, formed `Λ(q)`, blended it in. It made oscillation measurably *worse*:
  std went from 29.5N (no decoupling) to 68.8N at full decoupling, with every
  blend fraction in between (10%, 25%, 50%...) monotonically worse than no
  decoupling at all. Root cause (inferred, not separately proven): `Λ(q)` is
  itself derived from the same `M(q)` that's changing shape as the arm swings
  through the sliding stroke, and computing/inverting it every control step
  introduces its own numerical sensitivity right at the frequencies where the
  contact is already marginal — so instead of "correcting" the effective gain,
  it added a second, faster-varying gain multiplying an already-oscillating
  error signal. Reverted to plain Jacobian-transpose control; left as a
  documented dead end in `step_task_space()`'s docstring so it isn't re-tried
  without new evidence.

- **Flattening the pad geometry.** Hypothesis: the pad's ellipsoid curvature was
  causing the contact normal direction to change rapidly as the pad rocked
  slightly during sliding, injecting high-frequency direction changes into the
  contact force. Tested by flattening the ellipsoid (reducing its z-extent
  further, moving it closer to disc-like). Made oscillation slightly *worse*,
  not better — this rules out pad curvature as a primary driver, and suggests
  (though wasn't separately confirmed) that a flatter pad may have *increased*
  the effective contact stiffness at a given penetration (less compliant
  "give" at the edges), working against the softness fix rather than helping it.

- **Stiffer/faster contact (the "opposite" resonance direction).** One
  hypothesis for the oscillation was a resonance between the arm's task-space
  control loop and the contact's own time constant — if that were the whole
  story, making the contact *stiffer and faster* (shorter `solref` time
  constant) should shift the contact's natural frequency away from the control
  loop's and also help. Tried `solref` time constants around 1-2ms. This
  violates MuJoCo's own documented guideline that `solref`'s time constant
  should be at least `2×timestep` (timestep here is 1ms) — below that, the
  implicit contact solver's discretization becomes inaccurate and produces
  numerical breakdown (garbage forces, not a real signal). So this wasn't a
  fair test of the resonance theory either way; it just demonstrates the
  contact can't safely be pushed stiffer than it already was at this timestep.

- **Widening contact margin.** `geom_margin` is MuJoCo's contact *detection*
  buffer — geoms are considered "in contact" (and solref/solimp start acting)
  once they're within `margin` of touching, before actual geometric
  penetration. The dropout failure mode observed was `ncon` (active contact
  count) hitting 0 for a step or more during sliding — i.e. the pad briefly
  loses contact entirely as it slides, then reacquires it, producing an
  impact-like force transient on reacquisition. Widening margin should reduce
  this by keeping the contact constraint "engaged" slightly before/after actual
  geometric touching. It worked exactly as intended for that specific failure
  mode: dropouts went from 944 to 0 as margin was swept from 1.5mm to 20mm.
  But at the larger margin values, a *different*, worse failure mode appeared:
  continuous large-amplitude oscillation even with zero dropouts (std 60-90N).
  Plausible mechanism: with contacts started with such a wide margin,
  significant force is now being applied against small/negative true geometric
  penetration, i.e. exactly the wide-`solimp`-width miscalibration seen in the
  `SOFT_PAD_SOLIMP_WIDTH=0.010` test failures (finding #2/#3-adjacent) — normal
  sliding penetration now sits inside a much wider, softer ramp-up region than
  intended, so the constraint force response to small position noise becomes
  larger and noisier rather than smaller. Not pursued as the primary fix since
  the softer-`solref` approach solved the same dropout problem without this
  side effect.

## 5. Device-side: haptic "jitter" after the contact fix

After the contact fix, sim-side force is well-behaved (0% of samples over the
break threshold, only 2.8% over the cap, no dropout events, in a real recorded
episode). But felt jitter was still significant. Analysis of the recorded episode
(`normal_force_n`, `haptic_force_sent`, `device_force_cmd`, `device_force_measured`,
`device_vel` fields) worked by comparing "roughness" (std of consecutive sample-
to-sample differences — a simple proxy for high-frequency content that doesn't
require picking an FFT window) at each stage of the force pipeline, in-contact:

| signal | roughness (N/step) | where it lives |
|---|---|---|
| `normal_force_n` (raw sim contact force) | 0.35 | MuJoCo contact solver output |
| `haptic_force_sent` (after `tau`/`rate` filtering) | 0.27 | Python control loop, filtered |
| `device_force_cmd` (what's actually written to hardware) | 0.45 | `fd_omega.py` servo thread |
| `device_force_measured` (hardware readback) | 0.46 | device firmware |

The filtering step (raw → sent) does what it should: roughness drops 0.35→0.27.
But `device_force_cmd` — computed *downstream* of the already-filtered `sent`
value — is *rougher* than its own input. Something between "sent" and "actually
written to the device" was adding noise back in. That pointed at `fd_omega.py`'s
velocity-damping term:

```python
f = elastic - damping_b * scale * vel   # fd_omega.py
```

- Isolated the term's contribution by reconstructing `sent + predicted_damping`
  from the logged `haptic_force_sent` and `device_vel` and comparing its
  roughness to the *actual* recorded `device_force_cmd`: predicted roughness
  0.67 vs. actual 0.45-0.46 — same order of magnitude, confirming the damping
  term (not something else downstream) explains most of the gap.
- Raw device velocity noise floor: per-axis step-to-step jumps of only 3–8 mm/s —
  genuinely small, this is not a "broken/loose hardware" signal.
- At `--damping 100` (raised earlier in this session, 60→100, to fight the
  now-fixed sim oscillation), that noise floor gets amplified into **0.3–0.85N of
  force noise per axis per millisecond** — the same order of magnitude as the felt
  jitter, and comparable to the *entire* sim-side steady-state force std (4.4N)
  during a real sanding stroke.
- Confirmed even in free space (no sim contact, `normal_force_n < 1N`):
  `device_force_cmd` still has mean ≈0.88N / std ≈2.26N per sample there, versus
  `haptic_force_sent`'s mean 0.11N / std 0.20N at the same samples — a signal
  that should be near-silent at zero commanded force is not, and the excess is
  entirely attributable to this one term (it isn't present upstream in `sent`).

**Not a sim problem, and not "overdamping" in the classical slow/sluggish sense** —
see the explanation below. Recommended fix: lower `--damping` back toward 40–60
now that the sim no longer needs the extra margin, and/or low-pass filter `vel`
before it enters the damping term.

## Why higher damping made jitter *worse*, not just slower

The "more damping = slower/sluggish, not unstable" intuition holds for continuous
damping acting on a *clean* velocity signal. Two things break that assumption here:

1. `vel` is a **noisy, sampled estimate** (differentiated discrete position at
   ~1kHz), not a clean physical quantity. Real mechanisms have some noise floor
   from quantization/transmission — here it's small (3–8 mm/s) but nonzero.
2. `damping_b` is a **multiplicative gain** on that noisy signal, not a filter.
   Whatever noise rides on `vel` shows up directly, scaled by `damping_b`, in the
   output force. Raising `damping_b` doesn't add "more friction" — it adds a
   bigger amplifier in front of a fixed noise source.

So two effects compete as damping increases: real benefit (resisting genuine,
large-scale hand motion — this is what fixed the original sliding oscillation),
and noise amplification (scales linearly with the same gain, present even when
you're barely moving). At `damping_b=100` the noise term dominates. The fix isn't
"use less damping and accept worse stability" — it's decoupling the two effects:
keep enough gain to resist real motion, but filter the noise out of `vel` before
it's multiplied by that gain.
