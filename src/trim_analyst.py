"""Sail trim analyst — generates upwind-trim commentary for a Swan 42.

Given per-sail aero numbers (camber %, draft %, twist °, entry / exit
angles), plus luff-bend profile, outputs an HTML block with:

  * Per-sail trim reading (jib / main) with interpretation of the
    numbers vs the 8-9 kt upwind target.
  * Harmony analysis — do the jib and main shapes complement each
    other? Check luff curves (mast bend vs forestay sag), leech
    profiles (slot opening), camber matching.
  * High-mode vs low-mode tuning advice.
  * Two sailmaker-actionable improvement notes per sail.

This module is rule-based (no LLM call) — it inspects the numbers and
selects commentary from curated templates tied to target ranges. Tone
is that of an experienced trimmer / coach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---- Target ranges for Swan 42, 8-9 kt TWS, upwind ---------------------
# Sources: generic performance-cruiser trim windows; refined for J/109
# style masthead rig with deck-swept jibs.

TARGETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "main": {
        "camber_pct": (11.0, 14.0),
        "draft_pct": (45.0, 52.0),
        "twist_top_vs_bottom": (6.0, 12.0),  # degrees
        "entry_deg": (5.0, 12.0),
        "exit_deg": (2.0, 7.0),
    },
    "J1.5": {  # medium-wind jib — deeper
        "camber_pct": (11.5, 14.5),
        "draft_pct": (38.0, 44.0),
        "twist_top_vs_bottom": (5.0, 10.0),
        "entry_deg": (6.0, 13.0),
        "exit_deg": (3.0, 8.0),
    },
    "J2": {    # slightly flatter / higher-range jib
        "camber_pct": (10.5, 13.5),
        "draft_pct": (37.0, 43.0),
        "twist_top_vs_bottom": (5.0, 10.0),
        "entry_deg": (5.0, 11.0),
        "exit_deg": (2.0, 7.0),
    },
}


@dataclass
class SailReading:
    """Raw numbers extracted from a sail's analysis pass."""
    name: str                     # "J2 Raving Swan", etc.
    sail_type: str                # "main", "J1.5", "J2"
    stripes_top_to_bottom: List[Dict[str, float]]  # each: camber, draft, twist, entry, exit
    luff_max_bend_pct: float      # max perpendicular deviation / chord %
    luff_max_bend_at_pct: float   # position of max along luff (0 tack → 100 head)
    leech_max_bend_pct: float
    chord_m_foot: Optional[float] = None


def _verdict(value: float, rng: Tuple[float, float]) -> str:
    lo, hi = rng
    if value < lo - (hi - lo) * 0.15:
        return "low"
    if value < lo:
        return "slightly-low"
    if value > hi + (hi - lo) * 0.15:
        return "high"
    if value > hi:
        return "slightly-high"
    return "in-range"


def _verdict_colour(v: str) -> str:
    return {
        "low": "#ff6b5a",
        "slightly-low": "#ffb020",
        "in-range": "#00d4aa",
        "slightly-high": "#ffb020",
        "high": "#ff6b5a",
    }.get(v, "#98a3b5")


def _avg_camber(reading: SailReading) -> float:
    return float(np.mean([s["camber_pct"] for s in reading.stripes_top_to_bottom]))


def _top_minus_bottom(reading: SailReading, key: str) -> float:
    if len(reading.stripes_top_to_bottom) < 2:
        return 0.0
    top = reading.stripes_top_to_bottom[0][key]
    bot = reading.stripes_top_to_bottom[-1][key]
    return float(top - bot)


# ---- Per-sail commentary -------------------------------------------------

def _format_stripe_row(s: Dict[str, float], pos: int) -> str:
    return (
        f"<tr><td>{pos}</td>"
        f"<td>{s['camber_pct']:.1f}%</td>"
        f"<td>{s['draft_pct']:.0f}%</td>"
        f"<td>{s['twist_deg']:+.1f}°</td>"
        f"<td>{s['entry_deg']:.1f}°</td>"
        f"<td>{s['exit_deg']:.1f}°</td></tr>"
    )


def _badge(label: str, verdict: str) -> str:
    clr = _verdict_colour(verdict)
    return (f"<span style='background:{clr}22;color:{clr};"
            f"padding:2px 8px;border-radius:4px;font-size:11px;"
            f"border:1px solid {clr}55;'>{label}: {verdict}</span>")


def _sail_block(reading: SailReading) -> str:
    tgt = TARGETS.get(reading.sail_type, TARGETS["main"])
    avg_c = _avg_camber(reading)
    draft_avg = float(np.mean([s["draft_pct"] for s in reading.stripes_top_to_bottom]))
    entry_avg = float(np.mean([abs(s["entry_deg"]) for s in reading.stripes_top_to_bottom]))
    twist_span = _top_minus_bottom(reading, "twist_deg")

    # Interpretation lines
    lines: List[str] = []
    lines.append(
        f"<b>{reading.name}</b> — "
        f"{reading.sail_type.upper()} · "
        f"{len(reading.stripes_top_to_bottom)} stripes analysed"
    )

    # Recap table
    table = (
        "<table class='stats' style='margin-top:4px'>"
        "<tr><th>#</th><th>camber</th><th>draft</th>"
        "<th>twist</th><th>entry</th><th>exit</th></tr>"
        + "".join(
            _format_stripe_row(s, i + 1)
            for i, s in enumerate(reading.stripes_top_to_bottom)
        )
        + "</table>"
    )

    # Verdicts
    v_camber = _verdict(avg_c, tgt["camber_pct"])
    v_draft = _verdict(draft_avg, tgt["draft_pct"])
    v_twist = _verdict(abs(twist_span), tgt["twist_top_vs_bottom"])
    badges = (
        f"<div style='margin:6px 0'>"
        f"{_badge(f'avg camber {avg_c:.1f}%', v_camber)} "
        f"{_badge(f'avg draft {draft_avg:.0f}%', v_draft)} "
        f"{_badge(f'twist top-bot {twist_span:+.1f}°', v_twist)}"
        f"</div>"
    )

    # Reading paragraph
    read: List[str] = []
    if reading.sail_type == "main":
        read.append(
            f"Mainsail average camber sits at <b>{avg_c:.1f}%</b>, target "
            f"{tgt['camber_pct'][0]:.0f}–{tgt['camber_pct'][1]:.0f}% for "
            f"8–9 kt upwind. "
        )
        if v_camber in ("low", "slightly-low"):
            read.append(
                "Too flat — you're bleeding drive in this breeze. "
                "Ease backstay, reduce outhaul, sheet the main harder "
                "to load the leech."
            )
        elif v_camber in ("high", "slightly-high"):
            read.append(
                "Over-powered for the wind range — pull the backstay, "
                "tension the outhaul, flatten the foot and pull the "
                "cunningham to move draft forward."
            )
        else:
            read.append("Depth is on target — shape is working for the range.")

        read.append(
            f" Draft position averages <b>{draft_avg:.0f}%</b> (target "
            f"{tgt['draft_pct'][0]:.0f}–{tgt['draft_pct'][1]:.0f}%). "
        )
        if draft_avg > tgt["draft_pct"][1]:
            read.append(
                "Draft is too far aft — use more cunningham, increase "
                "backstay slightly to flatten entry."
            )
        elif draft_avg < tgt["draft_pct"][0]:
            read.append(
                "Draft too far forward — ease cunningham, loosen halyard."
            )

        # Luff / mast-bend commentary
        read.append(
            f"<br/>Main <b>luff bend</b> peaks at "
            f"<b>{reading.luff_max_bend_pct:.2f}%</b> of luff length — "
        )
        if reading.luff_max_bend_pct < 0.8:
            read.append(
                "mast is running straight. For 8–9 kt, add a touch of "
                "prebend to match the sail's built-in luff round, "
                "otherwise the main will be too full in the middle."
            )
        elif reading.luff_max_bend_pct < 1.8:
            read.append(
                "mast bend is moderate — a reasonable match to sail luff "
                "curve. Watch for inversion at the top."
            )
        else:
            read.append(
                "mast is heavily bent — check the top of the main isn't "
                "over-flattened and losing drive."
            )

    else:  # jib
        read.append(
            f"{reading.sail_type} average camber <b>{avg_c:.1f}%</b> vs "
            f"target {tgt['camber_pct'][0]:.0f}–{tgt['camber_pct'][1]:.0f}%. "
        )
        if v_camber in ("low", "slightly-low"):
            read.append(
                "Too flat — ease the halyard slightly and let the forestay "
                "sag to open the shape. You want power in this range."
            )
        elif v_camber in ("high", "slightly-high"):
            read.append(
                "Too deep — add halyard, tighten forestay (more runner / "
                "backstay) to flatten and open the leech."
            )
        else:
            read.append("Depth on target — good power for the breeze.")

        read.append(
            f" Draft at <b>{draft_avg:.0f}%</b> of chord. "
        )
        if draft_avg > tgt["draft_pct"][1]:
            read.append(
                "Draft too far aft — add halyard and tension the luff."
            )
        elif draft_avg < tgt["draft_pct"][0]:
            read.append(
                "Draft too far forward — ease halyard; the luff is "
                "over-tensioned."
            )

        read.append(
            f"<br/><b>Forestay / luff sag</b> peaks at "
            f"<b>{reading.luff_max_bend_pct:.2f}%</b> of the luff — "
        )
        if reading.luff_max_bend_pct < 1.0:
            read.append(
                "forestay tight — good for upwind control but loses "
                "power; ease runners slightly to let the luff sag ~1.5% "
                "in this range."
            )
        elif reading.luff_max_bend_pct < 2.5:
            read.append("sag is in the sweet spot for 8–9 kt upwind.")
        else:
            read.append(
                "too much sag — tighten the runner / backstay to flatten "
                "the entry or you'll over-round the luff and stall."
            )

    read.append(f"<br/>Twist (top vs bottom chord) spans "
                  f"<b>{twist_span:+.1f}°</b> — ")
    if abs(twist_span) < tgt["twist_top_vs_bottom"][0]:
        read.append(
            "under-twisted; the top is stalled. Ease sheet, drop "
            "traveler down (main) or car back (jib)."
        )
    elif abs(twist_span) > tgt["twist_top_vs_bottom"][1]:
        read.append(
            "over-twisted; the top is spilling. Sheet harder, car "
            "forward (jib) or traveler up (main)."
        )
    else:
        read.append("twist is in range — top and bottom are matched.")

    return (
        f"<div class='sail-card' style='background:#0f162b;"
        f"border:1px solid #1b2640;border-radius:8px;padding:14px;"
        f"margin:10px 0;'>"
        f"{lines[0]}{badges}{table}"
        f"<p style='line-height:1.55;margin-top:10px'>{''.join(read)}</p>"
        f"</div>"
    )


# ---- Harmony analysis ---------------------------------------------------

def _harmony_block(readings: List[SailReading]) -> str:
    main = next((r for r in readings if r.sail_type == "main"), None)
    jibs = [r for r in readings if r.sail_type.startswith("J")]
    if main is None or not jibs:
        return ""
    avg_main_c = _avg_camber(main)
    lines: List[str] = []
    for j in jibs:
        avg_j = _avg_camber(j)
        delta = avg_main_c - avg_j
        if abs(delta) < 1.0:
            verdict = "matched"
            clr = "#00d4aa"
        elif delta > 0:
            verdict = f"main {delta:+.1f}% deeper than {j.sail_type}"
            clr = "#ffb020"
        else:
            verdict = f"{j.sail_type} {-delta:+.1f}% deeper than main"
            clr = "#ffb020"
        lines.append(
            f"<li><b>{j.name}</b> vs main: "
            f"<span style='color:{clr}'>{verdict}</span>.  "
            f"Slot is closed if the jib leech twist is less than the main "
            f"leech twist (check J{j.sail_type[-1] if j.sail_type[-1].isdigit() else '?'} "
            f"lower-stripe exit angle vs the main's lower-stripe exit).</li>"
        )

    # Forestay/mast harmony
    lines.append(
        f"<li><b>Mast bend vs forestay sag</b>: main luff bends "
        f"{main.luff_max_bend_pct:.2f}%, "
        + ", ".join(f"{j.sail_type} sags {j.luff_max_bend_pct:.2f}%" for j in jibs)
        + (". The jib luff sag should be <i>slightly greater</i> than mast "
           "bend — otherwise the rig's forward stiffness overpowers the "
           "jib shape. Target jib-sag / mast-bend ratio ≈ 1.2–1.5 in "
           "8–9 kt.</li>")
    )

    return (
        f"<h3 style='color:#e4eaf2'>Harmony between sails</h3>"
        f"<ul style='line-height:1.55'>{''.join(lines)}</ul>"
    )


# ---- High vs low mode advice --------------------------------------------

def _mode_block() -> str:
    return """
<h3 style='color:#e4eaf2'>High-mode vs Low-mode upwind (8–9 kt TWS)</h3>
<div style='display:flex;gap:12px;flex-wrap:wrap;'>
  <div style='flex:1;min-width:280px;background:#0f162b;
              border:1px solid #1b2640;border-radius:8px;padding:12px;'>
    <b>High mode — point higher, slower VMG trade</b>
    <ul style='line-height:1.55;font-size:13px;'>
      <li>Flatten both sails: ease backstay 10–15 %, pull outhaul tight, add halyard</li>
      <li>Traveler down the track so main twists open at the top</li>
      <li>Jib car back ½ hole — opens the leech, flattens the top</li>
      <li>Forestay: tight (runner on) — entry flat, less forgiving but higher pointing</li>
      <li>Steer 2–3° higher than the groove — feather on the puffs</li>
    </ul>
  </div>
  <div style='flex:1;min-width:280px;background:#0f162b;
              border:1px solid #1b2640;border-radius:8px;padding:12px;'>
    <b>Low mode — foot off, speed build</b>
    <ul style='line-height:1.55;font-size:13px;'>
      <li>Power up: ease halyards, ease backstay, ease outhaul</li>
      <li>Traveler up — centre the boom, load the leech</li>
      <li>Jib car forward ½ hole — closes the leech, deepens top</li>
      <li>Let the forestay sag (~2 %) — rounder entry, more grunt</li>
      <li>Bow down 3–5°, build target boat-speed, then squeeze up</li>
    </ul>
  </div>
</div>
"""


# ---- Sailmaker notes ----------------------------------------------------

def _sailmaker_notes(readings: List[SailReading]) -> str:
    blocks: List[str] = []
    for r in readings:
        notes: List[str] = []
        avg_c = _avg_camber(r)
        avg_draft = float(np.mean([s["draft_pct"] for s in r.stripes_top_to_bottom]))
        top = r.stripes_top_to_bottom[0]
        bot = r.stripes_top_to_bottom[-1]

        if r.sail_type == "main":
            # Check mast-bend / luff-curve match
            if r.luff_max_bend_pct < 1.0:
                notes.append(
                    "Main shows almost no luff bend projection — the "
                    "built-in luff round might be <b>insufficient</b> for "
                    "this rig; adding ~20 mm of luff round in the middle "
                    "would let the crew induce cleaner mast bend without "
                    "over-flattening."
                )
            elif top["camber_pct"] > 14.0:
                notes.append(
                    "<b>Top section is too full</b> in 8–9 kt — consider "
                    "recutting the top panel to remove 10–15 mm of depth, "
                    "or adjust the luff curve so the top flattens more as "
                    "the backstay loads."
                )
            # Twist pattern
            twist_span = abs(top["twist_deg"] - bot["twist_deg"])
            if twist_span < 4.0:
                notes.append(
                    "Leech is <b>not releasing enough at the head</b> "
                    "(low twist gradient). Leech hollow / batten stiffness "
                    "could be adjusted so the top panel opens 2–3° more."
                )
            else:
                notes.append(
                    "Bottom-section exit angle is a bit sharp — possibly "
                    "too much shelf depth or outhaul-dependent seam shape; "
                    "consider a flatter foot panel for upwind-dominant use."
                )

        else:  # jib
            label = r.sail_type
            if r.sail_type == "J2":
                if avg_c > 13.5:
                    notes.append(
                        "<b>{label} is deeper than its design slot</b> — "
                        "this is a higher-range jib and it should flatten "
                        "near the forestay. Recheck the broad-seam overlap "
                        "on the middle panels.".replace("{label}", label)
                    )
                else:
                    notes.append(
                        "Camber is OK for a J2. Check the <b>entry radius</b> — "
                        "the top stripe is showing a slightly sharp entry "
                        "angle which suggests the luff curve could gain 10 mm "
                        "in the top third for a smoother groove."
                    )
                notes.append(
                    "Leech shape max bend at <b>"
                    f"{r.leech_max_bend_pct:.1f}%</b> of chord — for a J2 "
                    "used in 12-15 kt targets, trim the hollow so the leech "
                    "projects less at full hoist."
                )

            elif r.sail_type == "J1.5":
                if bot["draft_pct"] > 45:
                    notes.append(
                        "<b>Bottom draft is too aft</b> for a J1.5 — "
                        "shelf / foot fullness is pushing the depth back. "
                        "Trim the foot panel by 5–8 mm of broad seam to "
                        "bring the draft forward to ~40 %."
                    )
                else:
                    notes.append(
                        "Bottom shape is in range. Watch the <b>entry "
                        f"angle at the head ({top['entry_deg']:.1f}°) — "
                        "too sharp will cause early luffing in puffs. "
                        "Consider flattening the top panel by 3–5 mm of "
                        "luff round."
                    )
                notes.append(
                    "Top stripe shows camber "
                    f"{top['camber_pct']:.1f}% — for a medium-wind J1.5 "
                    "aim for 12–14 % there; adjust top panel depth if "
                    "it's drifting outside that window."
                )

        blocks.append(
            f"<div class='sail-card' style='background:#0f162b;"
            f"border:1px solid #1b2640;border-radius:8px;padding:14px;"
            f"margin:10px 0;'>"
            f"<b>{r.name}</b> · {r.sail_type.upper()} — "
            "<span style='color:#98a3b5;font-size:11px'>2 sailmaker "
            "action points</span>"
            "<ol style='line-height:1.55;margin-top:4px'>"
            + "".join(f"<li>{n}</li>" for n in notes[:2])
            + "</ol></div>"
        )

    return (
        "<h3 style='color:#e4eaf2'>Sailmaker action points</h3>"
        + "".join(blocks)
    )


# ---- Top-level generator ------------------------------------------------

def build_comments_html(readings: List[SailReading]) -> str:
    intro = (
        "<div style='background:#0f162b;border:1px solid #1b2640;"
        "border-radius:8px;padding:14px;margin:10px 0;line-height:1.55;'>"
        "<h2 style='color:#e4eaf2;margin-top:0'>Trim analyst — Swan 42, "
        "8–9 kt TWS, upwind</h2>"
        "<p>Boat is in the <b>transition</b> wind band for a Swan 42: "
        "below target speed you need power (fuller, more twist-closed "
        "sails); above target you want pointing (flatter, twist-opened). "
        "The readings below benchmark each sail against the 8–9 kt upwind "
        "window and call out what to move.</p>"
        "</div>"
    )
    per_sail = (
        "<h3 style='color:#e4eaf2'>Per-sail reading</h3>"
        + "".join(_sail_block(r) for r in readings)
    )
    return intro + per_sail + _harmony_block(readings) + _mode_block() + _sailmaker_notes(readings)
