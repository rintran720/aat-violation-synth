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
    return (f"Each new LSP has EXACTLY the size of the {original} in image 1: the same length, width and thickness, "
            "the same orientation (its sides parallel to the original sheet's) and the same perspective at that "
            "spot, so the plates look like identical sheets laid end to end; it is not smaller, not larger, not "
            f"wider and not deeper. Take its size from the {original} in image 1, never from image 2.")


def copy_cargo(height_of: str) -> str:
    return ("The cargo copies the real loads already in image 1: the same kind of load (for example film-wrapped, a "
            "carton stack, a crate), the same look, colour and wrapping, and about the same height as "
            f"{height_of}, with the same CCTV blur, noise and lighting as those loads, so it looks like one more of "
            "the loads that are already in the warehouse.")


KEEP_REST = "Every other forklift, LSP and load stays as it is."
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
                        "LSP (like image 2) lies flat directly in front of the loaded one in the direction the forks "
                        "point, its rear edge touching the loaded sheet's front edge along its full width. "
                        f"{same_size(LOADED)} On the new LSP stands one skid (like image 3) carrying one cargo load, "
                        f"centred on the sheet. {copy_cargo(ON_FORKLIFT)} Both LSPs must be clearly visible as two "
                        f"distinct flat plates, with a natural contact shadow. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes TWO LSPs at the same time: a second LSP "
                        "(like image 2) lies flat directly in front of it in the direction the forks point, its rear "
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
                        "LSPs (like image 2) lie flat in one straight row directly in front of the loaded one in the "
                        "direction the forks point, each new sheet's rear edge touching the front edge of the sheet "
                        "behind it along its full width, so the three sheets form one chain laid end to end. "
                        f"{same_size(LOADED)} On each new LSP stands one skid (like image 3) carrying one cargo load, "
                        f"centred on its sheet. {copy_cargo(ON_FORKLIFT)} The chain stays straight; where the floor "
                        "or the frame ends, the front sheet may be cut by the frame edge. All three LSPs must be "
                        f"clearly visible as three distinct flat plates, with natural contact shadows. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes THREE LSPs at the same time: two more LSPs "
                        "(like image 2) lie flat in one straight row directly in front of it in the direction the "
                        "forks point, each new sheet's rear edge touching the front edge of the sheet behind it along "
                        f"its full width, so the three sheets form one chain laid end to end. {same_size(EMPTY)} On "
                        "each new LSP stands one skid (like image 3) carrying one cargo load, centred on its sheet. "
                        f"{KEEP_EMPTY}{copy_cargo(IN_FRAME)} The chain stays straight; "
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
                        "the forklift that carries the loaded LSP now pushes TWO LSPs at the same time: a second LSP "
                        "(like image 2) lies flat directly in front of the loaded one in the direction the forks "
                        "point, its rear edge touching the loaded sheet's front edge along its full width. "
                        f"{same_size(LOADED)} The new LSP stays EMPTY: no skid, no cargo and nothing else on it, its "
                        "whole worn top surface visible. The load stays only on the original LSP. Both LSPs must be "
                        f"clearly visible as two distinct flat plates, with a natural contact shadow. {KEEP_REST}"),
                    "forklift-with-lsp-empty": (
                        "the forklift that pushes the empty LSP now pushes TWO EMPTY LSPs at the same time: a second "
                        "LSP (like image 2) lies flat directly in front of it in the direction the forks point, its "
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
            "carry_2_skids_side_by_side": {
                "title": "Carry 2 SKIDs side by side",
                "catalogue": "-",
                "status": "draft",
                "refs": ["SKID"],
                "changes": {
                    "forklift-empty": (
                        "the forklift with the empty forks is now carrying TWO separate skids (like image 2) at the "
                        "same time, placed side by side horizontally across its forks, each with one cargo load on "
                        f"it. {copy_cargo(IN_FRAME)} Both skids rest on the forks at the same height and must be "
                        "clearly visible as two distinct loaded pallets with a small gap between them, together "
                        f"wider than the forklift. No LSP. {KEEP_REST}"),
                    "forklift-with-cargo-no-lsp": (
                        "the forklift that carries one cargo load on its forks is now carrying TWO separate skids at "
                        "the same time: next to the load it carries, a second skid (like image 2) with one more cargo "
                        "load rests on the forks, the two placed side by side horizontally across the forks. "
                        f"{copy_cargo(ON_FORKLIFT)} Both skids rest on the forks at the same height and must be "
                        "clearly visible as two distinct loaded pallets with a small gap between them, together "
                        f"wider than the forklift. No LSP. {KEEP_REST}"),
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
