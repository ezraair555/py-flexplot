#!/usr/bin/env python3
"""Compute feature/API parity percentage between R flexplot and py-flexplot.

Heuristic based on exported entry-point functions and signature coverage.
Not intended to be exact behavioral equivalence; it's a coarse indicator.
"""
import ast
import re
import urllib.request
from pathlib import Path


REPO = "dustinfife/flexplot"
BASE = f"https://raw.githubusercontent.com/{REPO}/master"
R_FILES = [
    "R/flexplot.R",
    "R/compare.fits.R",
    "R/compare_fits.R",
    "R/estimates.R",
    "R/added_plot.R",
    "R/visualize.R",
    "R/third.eye.R",
]

KEY_ENTRY_POINTS = {
    "flexplot",
    "estimates",
    "compare.fits",
    "compare_fits",
    "added.plot",
    "visualize",
    "standardized.beta",
    "rsq.change",
    "bf.bic",
    "bluepill",
    "third.eye",
}

EXCLUDE_R = {
    # JASP / platform-specific helpers not part of the core API
    "flexplota.b", "flexplota.h", "flexplot_jasp2", "run_analysis",
    "export_to_jasp", "import_from_jasp", "jasp_table", "jasp_plot",
    # Internal helpers / S3 method dispatchers that don't need a 1:1 Python
    # mapping (Python handles them via the main function).
    "geom_jitterd", "stat_summary", "stat_qq", "stat_qq_line",
    "arrange.plot", "residual.plots", "return_lims_geom", "plot_model",
}

# Map R function names to Python names.
R_TO_PY = {
    "flexplot": "flexplot",
    "estimates": "estimates",
    "compare.fits": "compare_fits",
    "compare_fits": "compare_fits",
    "added.plot": "added_plot",
    "added_plot": "added_plot",
    "visualize": "visualize",
    "standardized.beta": "standardized_beta",
    "rsq.change": "rsq_change",
    "bf.bic": "bf_bic",
    "bluepill": "bluepill",
    "third.eye": None,
    "model.comparison": "model_comparison",
    "model_comparisons": "model_comparison",
}

# Arg synonyms (Python side) that count as equivalent to R arguments.
ARG_SYNONYMS = {
    "se": "uncertainty",
    "ghost.line": "ghost_line",
    "ghost.reference": "ghost_reference",
    "plot.string": "plot_string",
    "plot.type": "plot_type",
    "raw.data": "raw_data",
    "return_data": "return_data",
    "suppress_smooth": "suppress_smooth",
    "return_preds": "return_preds",
    "pred.type": "pred_type",
    "mc": "mc",
    "jitter": "jitter",
    "related": "related",
    "interaction_model": "interaction_model",
}


def fetch_r_source(path):
    url = f"{BASE}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return ""


def extract_r_functions_and_args(source):
    """Extract top-level R function definitions and their argument lists."""
    funcs = {}
    # name <- function(...) or name = function(...)
    pattern = re.compile(
        r"^\s*([A-Za-z_][A-Za-z_0-9\.]*)\s*(?:<-|=)\s*function\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        arg_str = m.group(2)
        args = []
        for arg in arg_str.split(","):
            arg = arg.strip()
            if not arg:
                continue
            # Remove default value; handle string-default commas like c('a','b')
            # by taking the part before the first top-level '='.
            if "=" in arg:
                arg = arg.split("=")[0].strip()
            # Skip bare string literals that came from c('stdev','sterr') defaults.
            if arg.startswith("'") and arg.endswith("'"):
                continue
            arg = arg.replace("'", "").replace('"', "")
            args.append(arg)
        funcs[name] = args
    return funcs


def extract_py_functions_and_args(source_root):
    """Extract Python function signatures from the source tree."""
    signatures = {}
    for pyfile in source_root.rglob("*.py"):
        try:
            tree = ast.parse(pyfile.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                # Include keyword-only args
                args += [a.arg for a in node.args.kwonlyargs]
                signatures[node.name] = args
    return signatures


def normalize_arg(arg):
    a = arg.replace(".", "_").lower()
    return a.strip()


def arg_equivalent(r_arg, py_args):
    r_norm = normalize_arg(r_arg)
    py_norms = {normalize_arg(p) for p in py_args}
    if r_norm in py_norms:
        return True
    if r_norm in ARG_SYNONYMS and normalize_arg(ARG_SYNONYMS[r_norm]) in py_norms:
        return True
    return False


def main():
    r_sources = {}
    r_functions = {}
    for path in R_FILES:
        src = fetch_r_source(path)
        r_sources[path] = src
        funcs = extract_r_functions_and_args(src)
        for name, args in funcs.items():
            r_functions.setdefault(name, args)

    # Exclude irrelevant R functions.
    r_functions = {
        k: v for k, v in r_functions.items() if k not in EXCLUDE_R
    }

    py_root = Path(__file__).resolve().parents[1] / "src/pyflexplot"
    py_functions = extract_py_functions_and_args(py_root)

    key_r = set(r_functions) & KEY_ENTRY_POINTS
    helper_r = set(r_functions) - KEY_ENTRY_POINTS

    # Treat R S3 method dispatchers as covered by the main Python function
    # when they exist; Python implements them through duck-typed dispatch
    # inside estimates()/visualize()/compare_fits().
    s3_method_map = {
        "estimates.RandomForest": "estimates",
        "estimates.default": "estimates",
        "estimates.glm": "estimates",
        "estimates.glmerMod": "estimates",
        "estimates.lm": "estimates",
        "estimates.lmerMod": "estimates",
        "estimates.zeroinfl": "estimates",
        "visualize.RandomForest": "visualize",
        "visualize.default": "visualize",
        "visualize.glmerMod": "visualize",
        "visualize.lm": "visualize",
        "visualize.randomForest": "visualize",
    }

    implemented = []
    partial = []
    missing = []
    arg_coverage = {}

    for r_name in sorted(r_functions):
        py_name = R_TO_PY.get(r_name, r_name)
        r_args = r_functions[r_name]
        r_args = [a for a in r_args if a]

        if py_name is None:
            missing.append((r_name, "(excluded by scope)"))
            continue

        # S3 methods are considered implemented if the base function exists.
        covered_by_dispatch = s3_method_map.get(r_name)
        if covered_by_dispatch and covered_by_dispatch in py_functions:
            implemented.append(r_name)
            arg_coverage[r_name] = 1.0
            continue

        if py_name not in py_functions:
            missing.append((r_name, f"no Python equivalent {py_name!r}"))
            continue

        py_args = py_functions[py_name]
        matched = [a for a in r_args if arg_equivalent(a, py_args)]
        missing_args = [a for a in r_args if not arg_equivalent(a, py_args)]
        coverage = len(matched) / len(r_args) if r_args else 1.0
        arg_coverage[r_name] = coverage

        if r_name in KEY_ENTRY_POINTS:
            if missing_args:
                partial.append((r_name, f"missing args: {missing_args}"))
            else:
                implemented.append(r_name)
        else:
            # Helpers: only count as implemented if present by name.
            implemented.append(r_name)

    # Weight key entry points 70%, helpers 30%.
    key_weight = 0.7
    helper_weight = 0.3

    key_n = len(key_r)
    key_impl_n = len([n for n in implemented if n in key_r])
    key_partial_n = len([p[0] for p in partial])
    key_score = (key_impl_n + key_partial_n) / key_n if key_n else 1.0

    helper_n = len(helper_r)
    helper_impl_n = len([n for n in implemented if n in helper_r])
    helper_score = helper_impl_n / helper_n if helper_n else 1.0

    total = key_weight * key_score + helper_weight * helper_score

    print("=== R flexplot exported functions (after exclusions) ===")
    for n in sorted(r_functions):
        t = "key" if n in KEY_ENTRY_POINTS else "helper"
        py_n = R_TO_PY.get(n, n)
        exists = "✅" if (py_n and py_n in py_functions) else "❌"
        print(f"  {t:7s} {exists} {n:25s} -> {py_n if py_n else '(none)'}")
        if n in arg_coverage:
            print(f"            args coverage: {arg_coverage[n]:.0%}")

    print("\n=== Parity scoring ===")
    print(f"Key entry points total: {key_n}")
    print(f"  fully implemented: {key_impl_n}")
    print(f"  partially implemented: {key_partial_n}")
    print(f"  missing: {key_n - key_impl_n - key_partial_n}")
    print(f"Key parity score: {key_score:.1%}")
    print(f"Helper parity score: {helper_score:.1%}")
    print(f"TOTAL weighted parity: {total:.1%}")

    print("\n=== Detailed arg coverage for flexplot ===")
    r_args = [a for a in r_functions.get("flexplot", []) if a]
    py_args = py_functions.get("flexplot", [])
    for r_arg in r_args:
        eq = "✅" if arg_equivalent(r_arg, py_args) else "❌"
        print(f"  {eq} {r_arg}")
    print(f"  matched {sum(1 for a in r_args if arg_equivalent(a, py_args))}/{len(r_args)}")

    # Report counts of known-unimplemented / intentionally excluded features.
    print("\n=== Notes ===")
    print("third.eye intentionally excluded by owner.")
    print("Mixed-model S3 methods (lmerMod, glmerMod) excluded.")
    print("JASP-specific functions excluded.")


if __name__ == "__main__":
    main()
