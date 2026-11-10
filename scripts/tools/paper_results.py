#!/usr/bin/env python3
"""Write in-text numbers as LaTeX macros to results/paper_numbers.tex (one macro per line,
replaced in place if the macro already exists)."""
import os
import re
from scripts.directories import paper_numbers


def write_macro(name, value, fmt='{:.1f}'):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    val = fmt.format(value) if not isinstance(value, str) else value
    line = f'\\newcommand{{\\{name}}}{{{val}}}\n'
    lines = []
    if os.path.exists(paper_numbers):
        lines = [l for l in open(paper_numbers) if not l.startswith(f'\\newcommand{{\\{name}}}')]
    lines.append(line)
    open(paper_numbers, 'w').writelines(sorted(lines))
