"""Structural check of the paper without a LaTeX engine (standard library only).

    python3 tools/validate_tex.py            # report; exit 1 on structural errors
    python3 tools/validate_tex.py --final    # also fail while any \\TODO / \\TBD remains

Checks: \\input tree resolves; braces balance; \\begin/\\end nest; every \\cite key is in
paper_refs.bib; every \\ref/\\cref target has a \\label (no duplicates); every
\\includegraphics file exists (files guarded by \\IfFileExists are optional); kit files
present; typical slips (unescaped % after a number, _ / # / ^ / & outside math or tables);
and a rough page estimate for the 8-page limit (text + figures + tables, references excluded).
The page estimate is only a guide (+-15%); compile on Overleaf for the real count.
"""
import argparse
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

errors, warnings = [], []


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line[:m.start()] if m else line)
    return "\n".join(out)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def resolve_inputs(path, seen, optional=False):
    """Return list of (file, raw_text) in reading order."""
    if not os.path.exists(path):
        if optional:
            return []
        errors.append(f"missing \\input file: {os.path.relpath(path, PAPER)}")
        return []
    raw = read(path)
    files = [(path, raw)]
    body = strip_comments(raw)
    guarded = set(re.findall(r"\\IfFileExists\{([^}]*)\}", body))
    for name in re.findall(r"\\input\{([^}]*)\}", body):
        f = name if name.endswith(".tex") else name + ".tex"
        full = os.path.join(PAPER, f)
        if full in seen:
            continue
        seen.add(full)
        files += resolve_inputs(full, seen, optional=(name in guarded or f in guarded))
    return files


def check_braces(name, text):
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                line = text[:i].count("\n") + 1
                errors.append(f"{name}:{line}: unmatched '}}'")
                depth = 0
    if depth:
        errors.append(f"{name}: {depth} unclosed '{{'")


def check_envs(name, text):
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", text):
        line = text[:m.start()].count("\n") + 1
        if m.group(1) == "begin":
            stack.append((m.group(2), line))
        else:
            if not stack:
                errors.append(f"{name}:{line}: \\end{{{m.group(2)}}} without \\begin")
            elif stack[-1][0] != m.group(2):
                errors.append(f"{name}:{line}: \\end{{{m.group(2)}}} closes \\begin{{{stack[-1][0]}}} "
                              f"from line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
    for env, line in stack:
        errors.append(f"{name}:{line}: \\begin{{{env}}} never closed")


def bib_keys(path):
    if not os.path.exists(path):
        errors.append(f"missing bibliography {os.path.basename(path)} (run tools/build_bib.py)")
        return set()
    return set(re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", read(path), re.M))


def pdf_aspect(path):
    """height/width of the first MediaBox in a PDF, or of a PNG."""
    try:
        if path.endswith(".png"):
            with open(path, "rb") as f:
                head = f.read(24)
            w, h = struct.unpack(">II", head[16:24])
            return h / w
        data = open(path, "rb").read(400000)
        m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", data)
        if m:
            x0, y0, x1, y1 = map(float, m.groups())
            return (y1 - y0) / (x1 - x0)
    except Exception:  # noqa: BLE001
        pass
    return 0.6


def remove_math(line):
    line = re.sub(r"\$[^$]*\$", " ", line)
    line = re.sub(r"\\\(.*?\\\)", " ", line)
    return line


def check_slips(name, text):
    in_math_env = False
    in_tab = name.startswith("tables" + os.sep) or name.startswith("tables/")  # row files live inside a tabular
    for k, line in enumerate(text.split("\n"), 1):
        if re.search(r"\\begin\{(align|equation)", line):
            in_math_env = True
        if re.search(r"\\end\{(align|equation)", line):
            in_math_env = False
            continue
        if re.search(r"\\begin\{tabular", line):
            in_tab = True
        if re.search(r"\\end\{tabular", line):
            in_tab = False
        if in_math_env:
            continue
        raw = line
        if re.search(r"\d(?<!\\)%", raw.split("%%")[0]) and not raw.lstrip().startswith("%"):
            # a number directly followed by an unescaped % is almost always a slip
            for m in re.finditer(r"(\d)%", raw):
                if raw[m.start() + 1 - 1:m.start() + 1] != "\\":
                    warnings.append(f"{name}:{k}: unescaped % after a number (text after it is a comment)")
                    break
        s = remove_math(strip_comments(raw))
        s = re.sub(r"\\(texttt|url|label|ref|cref|Cref|cite[tp]?|input|includegraphics|IfFileExists|eqref|"
                   r"bibliography|bibliographystyle|graphicspath)(\[[^\]]*\])?\{[^}]*\}", " ", s)
        s = re.sub(r"\\csname @@input\\endcsname\s*\S+", " ", s)   # primitive \input (no file hooks) inside a tabular
        if re.search(r"(?<!\\)_", s):
            warnings.append(f"{name}:{k}: '_' outside math/texttt")
        if re.search(r"(?<!\\)#", s) and "\\newcommand" not in s and "\\def" not in s:
            warnings.append(f"{name}:{k}: unescaped '#'")
        if re.search(r"(?<!\\)\^", s) and "\\textsuperscript" not in s:
            warnings.append(f"{name}:{k}: '^' outside math")
        if not in_tab and re.search(r"(?<!\\)&", s):
            warnings.append(f"{name}:{k}: '&' outside a tabular")


def words_of(text):
    t = strip_comments(text)
    t = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{(table|figure)\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\(cite|label|ref|cref|Cref|eqref|input|includegraphics)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$\\]", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/]*", t))


def float_lines(text, figdirs):
    """Estimate column-lines used by figures and tables (one column ~ 53 lines of 12 pt)."""
    total = 0.0
    details = []
    body = strip_comments(text)
    # when an \IfFileExists offers two versions of a float, count only the larger one later;
    # here we simply count the first branch of each \IfFileExists pair
    body = re.sub(r"\\IfFileExists\{[^}]*\}\{%?(.*?)\n\}\{%?(.*?)\n\}", lambda m: m.group(2), body, flags=re.S)
    for m in re.finditer(r"\\begin\{(figure|table)(\*?)\}(.*?)\\end\{\1\2\}", body, flags=re.S):
        kind, star, inner = m.group(1), m.group(2), m.group(3)
        colw = 6.875 if star else 3.25
        lines = 0.0
        if kind == "figure":
            for g in re.finditer(r"\\includegraphics\[([^\]]*)\]\{([^}]*)\}", inner):
                wm = re.search(r"width=([\d.]*)\\(linewidth|textwidth)", g.group(1))
                frac = float(wm.group(1) or 1) if wm else 1.0
                f = g.group(2)
                path = next((os.path.join(d, f) for d in figdirs if os.path.exists(os.path.join(d, f))), None)
                asp = pdf_aspect(path) if path else 0.6
                lines += frac * colw * asp * 72.27 / 12
            fb = re.search(r"\\parbox\[c\]\[([\d.]+)in\]", inner)
            if fb:
                lines += float(fb.group(1)) * 72.27 / 12
        else:
            rows = inner.count("\\\\")
            lines += rows * 0.85 + 1.5
        cap = re.search(r"\\caption\{(.*)\}\s*\\label", inner, flags=re.S)
        capw = words_of(cap.group(1)) if cap else 0
        lines += capw / (24 if star else 12) * 0.9 + 1.5
        if star:
            lines *= 2  # spans both columns
        total += lines
        details.append((kind + star, round(lines, 1)))
    return total, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    main_tex = os.path.join(PAPER, "main.tex")
    files = resolve_inputs(main_tex, {main_tex})
    for req in ("wacv.sty", "ieeenat_fullname.bst"):
        if not os.path.exists(os.path.join(PAPER, req)):
            errors.append(f"missing kit file {req}")
    keys = bib_keys(os.path.join(PAPER, "paper_refs.bib"))
    labels, refs, cites = {}, [], []
    figdirs = [os.path.join(PAPER, "figs"), PAPER]
    todo_count = {}
    page_text_words, float_total, float_details = 0, 0.0, []
    for path, raw in files:
        name = os.path.relpath(path, PAPER)
        text = strip_comments(raw)
        check_braces(name, text)
        check_envs(name, text)
        check_slips(name, raw)
        guarded = set(re.findall(r"\\IfFileExists\{([^}]*)\}", text))
        for m in re.finditer(r"\\label\{([^}]*)\}", text):
            labels.setdefault(m.group(1), []).append(name)
        for m in re.finditer(r"\\(?:ref|cref|Cref|eqref|autoref)\{([^}]*)\}", text):
            refs += [(r.strip(), name) for r in m.group(1).split(",")]
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", text):
            cites += [(c.strip(), name) for c in m.group(1).split(",")]
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
            f = m.group(1)
            if not any(os.path.exists(os.path.join(d, f)) for d in figdirs):
                if ("figs/" + f) in guarded or f in guarded:
                    continue
                errors.append(f"{name}: missing figure {f}")
        todo_count[name] = (len(re.findall(r"\\TODO\{", text)), len(re.findall(r"\\TBD", text)))
        if name.startswith("sections/"):
            page_text_words += words_of(raw)
            ft, fd = float_lines(raw, figdirs)
            float_total += ft
            float_details += [(name,) + d for d in fd]
    # duplicate labels: the two \IfFileExists branches legitimately share a label
    for lab, where in labels.items():
        if len(where) > 1:
            txt = "".join(r for p, r in files if os.path.relpath(p, PAPER) == where[0])
            if txt.count("\\label{" + lab + "}") > 1 and "\\IfFileExists" in txt:
                continue
            errors.append(f"duplicate label {lab} in {where}")
    for r, name in refs:
        if r not in labels:
            errors.append(f"{name}: \\ref to undefined label {r}")
    missing = sorted({c for c, _ in cites if c not in keys})
    for c in missing:
        errors.append(f"citation key not in paper_refs.bib: {c}")
    unused = sorted(set(labels) - {r for r, _ in refs})

    # page estimate: 53 lines per column, 106 per page; ~11 words per text line;
    # title/author block ~26 column-lines; each section/paragraph heading ~1.5 lines
    heads = sum(len(re.findall(r"\\(sub)?section\{|\\paragraph\{", strip_comments(r))) for p, r in files)
    text_lines = page_text_words / 10.5
    est_lines = text_lines + float_total + 26 + heads * 1.6
    pages = est_lines / 106

    print(f"files: {len(files)} ({', '.join(os.path.relpath(p, PAPER) for p, _ in files)})")
    print(f"citations: {len(cites)} uses, {len({c for c, _ in cites})} distinct keys, {len(missing)} missing; "
          f"bib has {len(keys)} entries")
    print(f"labels: {len(labels)}, refs: {len(refs)}, unreferenced labels: {unused}")
    print("TODO / TBD markers per file:")
    for k, (t, b) in todo_count.items():
        if t or b:
            print(f"  {k:32s} TODO {t:2d}   TBD {b:3d}")
    print(f"page estimate (references excluded): {pages:.1f} of 8 "
          f"[text {page_text_words} words ~{text_lines / 106:.1f} p; floats ~{float_total / 106:.1f} p; "
          f"headings {heads}]")
    for d in float_details:
        print(f"    float {d[0]:28s} {d[1]:6s} ~{d[2]:5.1f} column-lines")
    if pages > 8.0:
        warnings.append(f"estimated length {pages:.1f} pages > 8 -- see the cut list in PLAN.md")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    tot_todo = sum(t + b for t, b in todo_count.values())
    if a.final and tot_todo:
        print(f"FINAL CHECK FAILED: {tot_todo} TODO/TBD markers remain")
        sys.exit(1)
    if errors:
        sys.exit(1)
    print("STRUCTURE OK" + ("" if not warnings else f" ({len(warnings)} warnings)"))


if __name__ == "__main__":
    main()
