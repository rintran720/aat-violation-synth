"""Violation engines, their cases and the input kinds each case takes, as the ONE change of the
try_astra_edit_batch prompt.

An engine is one violation class of the target system (docs/violation-engines.md). An input kind says what a valid
frame shows (INPUT_KINDS; inputs are sorted into folders named by it), and each kind feeds one engine. A case is one
way to turn a frame of a kind into that engine's violation with a single Astra edit: `changes` holds the ONE change
per input kind it takes, and `refs` the catalogue references that go in after the frame (images 2, 3, ...).
`catalogue` is the scenario id in config/scenarios.json and `status` its rule status there (pending_rule: whether
it counts as a violation is still to confirm; draft: a case of ours, not in the catalogue yet).
"""

INPUT_KINDS = {
    "forklift-with-lsp-cargo": {
        "title": "Forklift with LSP and cargo",
        "scene": "a counterbalance forklift carrying ONE LSP loaded with cargo on its forks",
        "engine": "forklift-pushing-multiple-lsps",
    },
    "forklift-with-lsp-empty": {
        "title": "Forklift with an empty LSP",
        "scene": "a counterbalance forklift pushing ONE EMPTY LSP (nothing on it) with its forks",
        "engine": "forklift-pushing-multiple-lsps",
    },
    "forklift-with-cargo-no-lsp": {
        "title": "Forklift with cargo, no LSP",
        "scene": "a counterbalance forklift carrying ONE cargo load (on a skid) on its forks, with no LSP",
        "engine": "forklift-charging-multiple-skids-horizontally",
    },
    "forklift-empty": {
        "title": "Forklift with empty forks",
        "scene": "a counterbalance forklift with EMPTY forks (nothing on them)",
        "engine": "forklift-charging-multiple-skids-horizontally",
    },
}


def same_size(original: str) -> str:
    """Where each new LSP goes and how big it is: against the sheet behind it, never over the original, and the
    original's size, never the cargo's (Astra drew new sheets on top of the original and shrank them to the load)."""
    return ("Placement: each new LSP lies flat on the floor right against the sheet behind it, their facing edges "
            "touching along the full width with no gap and no overlap. A new LSP never lies on, covers, overlaps or "
            f"hides any part of the {original}, which stays whole and fully visible exactly as in image 1. "
            f"Size: each new LSP has about the size of the {original} in image 1: the same length, width and "
            "thickness, the same orientation (its sides in line with the original sheet's) and the same perspective "
            "at that spot, so the plates look like identical sheets laid end to end; it is not smaller, not larger, "
            f"not wider and not deeper. Take its size from the {original} in image 1, never from image 2 and never "
            "from the cargo: the sheet keeps its full size even where the cargo on it is smaller, so its bare deck "
            "shows around the cargo.")


def copy_cargo(height_of: str) -> str:
    return ("The cargo copies the real loads already in image 1: the same kind of load (for example film-wrapped, a "
            "carton stack, a crate), the same look, colour and wrapping, and about the same height as "
            f"{height_of}, with the same CCTV blur, noise and lighting as those loads, so it looks like one more of "
            "the loads that are already in the warehouse.")


KEEP_REST = "Every other forklift, LSP and load stays as it is."
# the frame's own LSP is the first reference for a new one; the catalogue render (image 2) only stands in for it
LIKE_FRAME_LSP = ("a copy of the LSP the forklift pushes in image 1 (the same colour, wear, edges, thickness and CCTV "
                  "look; use image 2 only if image 1 showed no LSP)")
# the skids engine: where loads sit on the forks (user, 2026-10-06)
AGAINST_MAST = ("Every load sits all the way back on the forks: its skid and its rear face are pushed right against the "
                "forklift's vertical mast, the upright fork carriage the forks hang from, with no gap between the load "
                "and the mast.")
ALONG_FORKS = ("Each skid is turned lengthwise along the forks: its longer side runs parallel to the two fork tines, "
               "pointing the same way as the forks.")
# full-size loads: on forklift-empty frames Astra drew small parcels next to the forks (user, 2026-10-06)
FULL_SIZE = ("Each load is a FULL-SIZE warehouse load, as big as the large loads standing in the warehouse: its skid is "
             "about 1.2 m x 1.0 m, wider than the two fork tines together, and the load on it is about 1.2-1.5 m "
             "tall, about as high as the forklift's overhead guard. It is never a small parcel or a low box that "
             "looks small next to the forks.")
# how forks carry a skid (user, 2026-10-06): the tines go into the lower skid, so they and their shadow are hidden
FORKS_IN_SKID = ("The two fork tines go INTO the lower skid, through the openings between its runners under the top "
                 "deck, the way a forklift really carries a skid: the tines are hidden inside the skid along their "
                 "whole length, no bare tine shows beside, in front of or under the load, and no shadow of the tines "
                 "shows on the floor; only the skid and the load's own contact shadow are seen.")
FORKS_IN_SKIDS = ("The two fork tines go INTO both skids, through the openings between their runners under the top "
                  "decks, the way a forklift really carries skids: the tines are hidden inside the skids along their "
                  "whole length, no bare tine shows beside, between, in front of or under the loads, and no shadow of "
                  "the tines shows on the floor; only the skids and the loads' own contact shadows are seen.")
# loads centred on the forks and wider than them (user, 2026-10-06: important for forklift-empty frames)
FIT_FORKS = ("IMPORTANT, centring and width: every load and its skid is centred exactly between the two fork tines: "
             "the centre line of the load lies on the centre line between the tines, with the same overhang on the "
             "left and on the right, never shifted towards one tine or hanging off one side. Every load is WIDER than "
             "the two tines together (wider than the outer edges of both tines and as wide as the fork carriage or "
             "more), so both tines sit under the load, well inside its left and right edges. Make it really wide: "
             "MINIMUM WIDTH: each load (skid and cargo) is AT LEAST as wide as the forklift itself, measured across "
             "the forklift from the outer side of its left wheel to the outer side of its right wheel (about "
             "1.2-1.5 m, wider than the mast and the overhead guard); it may be a little wider, never narrower. Seen "
             "from the camera, the load's left and right sides line up with or reach past the forklift body's left "
             "and right sides, it is clearly broader than it is deep along the forks, and it never looks like a "
             "narrow box sitting between the tines.")
ACROSS_FORKS = ("Each skid and its load is turned crosswise: its longer side runs across both fork tines, at right "
                "angles to them, so the load spans the two forks.")
# two cargo loads one above the other on the forks (the skids engine's stacked case)
STACK = ("The upper skid sits squarely and level on the top of the lower load, aligned with it on every side and not "
         "overhanging, so the stack stands straight and stable; the two loads have about the same footprint, and "
         "together they are about twice as tall as one load.")
STACK_LOOK = ("Both loads must be clearly visible as two distinct loads, one above the other, with the upper skid's "
              "boards showing as a thin dark band between them, and the same lighting and shadows as the rest of the "
              "frame.")
# V2's three-sheet row (user, 2026-10-06): Astra left gaps between the sheets or slid one sheet into another
TIGHT_ROW = ("Layout of the row: picture the original sheet as one tile and lay two more identical tiles end to end "
             "in front of it, like three floor tiles in a line. Each sheet's front edge and the next sheet's rear "
             "edge are the SAME line on the floor: they meet exactly, with zero floor showing between them (not even "
             "a hand's width) and zero overlap. No sheet is pushed under, on top of or inside another sheet, none is "
             "shortened or partly hidden by its neighbour, and none is shifted sideways: the left edges of all three "
             "form one straight line and so do the right edges. Seen from the camera the row is three complete, "
             "equal rectangles in a line, each showing its own four corners and its own thin dark side edge, the only "
             "boundary between two sheets being one thin seam line. Measured along the forks the whole row is "
             "exactly three sheet lengths long: not longer (that would mean gaps) and not shorter (that would mean "
             "overlap). The cargo on each sheet stays inside that sheet's edges and does not bridge two sheets.")
LOADED, EMPTY = "loaded LSP", "empty LSP the forklift pushes"
ON_FORKLIFT, IN_FRAME = "the load on the forklift", "the loads in image 1"
KEEP_EMPTY = ("The original LSP the forklift pushes stays EMPTY, exactly as in image 1: add nothing on it. ")

ENGINES = {
    "forklift-pushing-multiple-lsps": {
        "name": "Forklift Pushing Multiple Lsps",
        "color": "#960399",
        "cases": {
            "push_2_lsp_cargo": {
                "title": "Push 2 LSP with Cargo",
                "catalogue": "V1",
                "status": "active",
                "refs": ["LSP", "SKID"],
                "changes": {
                    "forklift-with-lsp-cargo": (
                        "the forklift that carries the loaded LSP now carries TWO LSPs at the same time: a second "
                        f"LSP, {LIKE_FRAME_LSP}, lies flat directly in front of the loaded one in the direction the forks "
                        "point, its rear edge touching the loaded sheet's front edge along its full width. "
                        f"{same_size(LOADED)} On the new LSP stands one skid (like image 3) carrying one cargo load, "
                        f"centred on the sheet. {copy_cargo(ON_FORKLIFT)} Both LSPs must be clearly visible as two "
                        f"distinct flat plates, with a natural contact shadow. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes TWO LSPs at the same time: a second LSP, "
                        f"{LIKE_FRAME_LSP}, lies flat directly in front of it in the direction the forks point, its rear "
                        "edge touching the original sheet's front edge along its full width. "
                        f"{same_size(EMPTY)} On the new LSP stands one skid (like image 3) carrying one cargo load, "
                        f"centred on the sheet. {KEEP_EMPTY}{copy_cargo(IN_FRAME)} Both LSPs must be clearly visible "
                        f"as two distinct flat plates, with natural contact shadows. {KEEP_REST}"),
                },
            },
            "push_3_lsp_cargo": {
                "title": "Push 3 LSP with Cargo",
                "catalogue": "V2",
                "status": "active",
                "refs": ["LSP", "SKID"],
                "changes": {
                    "forklift-with-lsp-cargo": (
                        "the forklift that carries the loaded LSP now pushes THREE LSPs at the same time: two more "
                        f"LSPs, each {LIKE_FRAME_LSP}, lie flat in one straight row directly in front of the loaded one in the "
                        "direction the forks point, each new sheet's rear edge touching the front edge of the sheet "
                        "behind it along its full width, so the three sheets form one chain laid end to end. "
                        f"{same_size(LOADED)} On each new LSP stands one skid (like image 3) carrying one cargo load, "
                        f"centred on its sheet. {copy_cargo(ON_FORKLIFT)} {TIGHT_ROW} The chain stays straight; where the floor "
                        "or the frame ends, the front sheet may be cut by the frame edge. All three LSPs must be "
                        f"clearly visible as three distinct flat plates, with natural contact shadows. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes THREE LSPs at the same time: two more LSPs, "
                        f"each {LIKE_FRAME_LSP}, lie flat in one straight row directly in front of it in the direction the "
                        "forks point, each new sheet's rear edge touching the front edge of the sheet behind it along "
                        f"its full width, so the three sheets form one chain laid end to end. {same_size(EMPTY)} On "
                        "each new LSP stands one skid (like image 3) carrying one cargo load, centred on its sheet. "
                        f"{KEEP_EMPTY}{copy_cargo(IN_FRAME)} {TIGHT_ROW} The chain stays straight; "
                        "where the floor or the frame ends, the front sheet may be cut by the frame edge. All three "
                        "LSPs must be clearly visible as three distinct flat plates, with natural contact shadows. "
                        f"{KEEP_REST}"),
                },
            },
            "push_2_lsp_empty_extra": {
                "title": "Push 2 LSP, extra LSP empty",
                "catalogue": "V6",
                "status": "pending_rule",
                "refs": ["LSP", "SKID"],
                "changes": {
                    "forklift-with-lsp-cargo": (
                        "the forklift that carries the loaded LSP now pushes TWO LSPs at the same time: a second LSP, "
                        f"{LIKE_FRAME_LSP}, lies flat directly in front of the loaded one in the direction the forks "
                        "point, its rear edge touching the loaded sheet's front edge along its full width. "
                        f"{same_size(LOADED)} The new LSP stays EMPTY: no skid, no cargo and nothing else on it, its "
                        "whole worn top surface visible. The load stays only on the original LSP. Both LSPs must be "
                        f"clearly visible as two distinct flat plates, with a natural contact shadow. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes TWO EMPTY LSPs at the same time: a second "
                        f"LSP, {LIKE_FRAME_LSP}, lies flat directly in front of it in the direction the forks point, its "
                        "rear edge touching the original sheet's front edge along its full width. "
                        f"{same_size(EMPTY)} The new LSP stays EMPTY: no skid, no cargo and nothing else on it, its "
                        f"whole worn top surface visible. {KEEP_EMPTY}Both LSPs must be clearly visible as two "
                        f"distinct flat plates, with natural contact shadows. {KEEP_REST}"),
                },
            },
        },
    },
    "forklift-charging-multiple-skids-horizontally": {
        "name": "Forklift Charging Multiple Skids Horizontally",
        "color": "#974935",
        "cases": {
            "carry_2_skids_side_by_side": {      # the id stays as stored jobs name it
                "title": "Carry 2 cargo side by side",
                "catalogue": "-",
                "status": "draft",
                "refs": ["SKID"],
                "changes": {
                    "forklift-empty": (
                        "the forklift with the empty forks is now carrying TWO separate cargo loads at the same "
                        "time, each standing on its own skid (like image 2). The two loaded skids sit one behind the "
                        "other in a row along the two fork tines: the first is pushed right against the forklift's "
                        "vertical mast, the upright fork carriage the forks hang from, and the second stands "
                        "directly in front of it, further out along the forks, its rear touching the first load's "
                        "front with no gap, in line with it and not shifted sideways. Together they cover the forks "
                        f"and reach beyond the fork tips. {FIT_FORKS} {FORKS_IN_SKIDS} {FULL_SIZE} "
                        f"{copy_cargo(IN_FRAME)} Both skids rest on the "
                        "forks at the same height and must be clearly visible as two distinct loaded skids, one "
                        f"behind the other. No LSP. {KEEP_REST}"),
                    "forklift-with-cargo-no-lsp": (
                        "the forklift that carries one cargo load on its forks is now carrying TWO separate cargo "
                        "loads at the same time, each standing on its own skid (like image 2; if the carried load "
                        "shows no skid under it, give it one). Next to the load it carries, across the width of the "
                        "fork carriage, a second skid with one more cargo load rests on the forks, the two side by "
                        f"side. {ALONG_FORKS} {AGAINST_MAST} {copy_cargo(ON_FORKLIFT)} Both skids rest on the forks at "
                        "the same height and must be clearly visible as two distinct loaded skids with a small gap "
                        f"between them, together wider than the forklift. No LSP. {KEEP_REST}"),
                },
            },
            "carry_2_cargo_stacked": {
                "title": "Carry 2 cargo stacked",
                "catalogue": "-",
                "status": "draft",
                "refs": ["SKID"],
                "changes": {
                    "forklift-empty": (
                        "the forklift with the empty forks is now carrying TWO cargo loads at the same time, stacked "
                        "one on top of the other: on the forks rests one skid (like image 2) with one cargo load, and "
                        "on top of that load stands a second skid (like image 2) with a second cargo load. "
                        f"{ACROSS_FORKS} {FIT_FORKS} {FORKS_IN_SKID} {AGAINST_MAST} {FULL_SIZE} {STACK} "
                        f"{copy_cargo(IN_FRAME)} {STACK_LOOK} No LSP. "
                        f"{KEEP_REST}"),
                    "forklift-with-cargo-no-lsp": (
                        "the forklift that carries one cargo load on its forks is now carrying TWO cargo loads at the "
                        "same time, stacked one on top of the other: the load it carries stays on the forks (on a "
                        "skid like image 2; give it one if none shows), and on top of it stands a second skid (like "
                        f"image 2) with a second cargo load. {ACROSS_FORKS} (Turn the carried load that way if it "
                        f"lies differently.) {AGAINST_MAST} {STACK} {copy_cargo(ON_FORKLIFT)} {STACK_LOOK} No LSP. "
                        f"{KEEP_REST}"),
                },
            },
        },
    },
}
DEFAULT_ENGINE, DEFAULT_CASE, DEFAULT_INPUT = "forklift-pushing-multiple-lsps", "push_2_lsp_cargo", "forklift-with-lsp-cargo"


def case(engine: str, case_id: str) -> dict:
    """One case of an engine; KeyError names what is unknown."""
    if engine not in ENGINES:
        raise KeyError(f"unknown engine {engine!r}")
    cases = ENGINES[engine]["cases"]
    if case_id not in cases:
        raise KeyError(f"engine {engine!r} has no case {case_id!r}")
    return cases[case_id]


def change(engine: str, case_id: str, input_kind: str) -> str:
    """The ONE change of a case for a frame of input_kind; KeyError when the case does not take that kind."""
    changes = case(engine, case_id)["changes"]
    if input_kind not in changes:
        raise KeyError(f"case {case_id!r} takes {', '.join(changes)} inputs, not {input_kind!r}")
    return changes[input_kind]


def catalogue() -> dict:
    """The input kinds, engines and cases without the prompt text, for a UI."""
    return {
        "input_kinds": [{"id": kind, "title": k["title"], "engine": k["engine"]} for kind, k in INPUT_KINDS.items()],
        "engines": [{"id": engine_id, "name": engine["name"], "color": engine["color"],
                     "input_kinds": [kind for kind, k in INPUT_KINDS.items() if k["engine"] == engine_id],
                     "cases": [{"id": case_id, "title": c["title"], "catalogue": c["catalogue"],
                                "status": c["status"], "input_kinds": list(c["changes"])}
                               for case_id, c in engine["cases"].items()]}
                    for engine_id, engine in ENGINES.items()],
    }
