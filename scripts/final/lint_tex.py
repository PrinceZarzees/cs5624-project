#!/usr/bin/env python3
"""Lightweight LaTeX audit: citations, missing inputs/figures, AI drafts."""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]; tex=(root/'report/main.tex').read_text()
out=['# E13 LaTeX lint','',f'- main.tex lines: {len(tex.splitlines())}']
# citation spacing
bad=[(i+1,l) for i,l in enumerate(tex.splitlines()) if re.search(r'[^\s~{\(]\\cite',l)]
out.append(f'- citations missing `~`: {len(bad)}')
for i,l in bad: out.append(f'  - line {i}: {l.strip()}')
# includes
refs=[]
for m in re.finditer(r'\\(?:input|includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}',tex):
    p=m.group(1); cands=[root/'report'/p,root/'report'/f'{p}.tex',root/'report'/f'{p}.pdf']
    refs.append((p,any(c.exists() for c in cands)))
out.append(f'- included files: {len(refs)}')
for p,ok in refs: out.append(f'  - {"OK" if ok else "MISSING"} {p}')
# AI drafts
drafts=[]; start=None
for i,l in enumerate(tex.splitlines(),1):
    if '>>> AI-DRAFT' in l: start=i
    if '<<< AI-DRAFT' in l and start: drafts.append((start,i)); start=None
out.append(f'- AI-DRAFT blocks: {len(drafts)}')
for a,b in drafts: out.append(f'  - lines {a}-{b}')

# optional compile/page check (tectonic + pypdf are not part of the repo)
tectonic = os.environ.get('TECTONIC_BIN') or shutil.which('tectonic')
if tectonic:
    build = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    try:
        proc = subprocess.run([tectonic, '-X', 'compile', 'main.tex', '--outdir', str(build)],
                              cwd=root/'report', capture_output=True, text=True, timeout=900, env=env)
        pdf = build/'main.pdf'
        out.append(f'- compile: returncode {proc.returncode}, pdf {"written" if pdf.exists() else "missing"}')
        errs = [l for l in (proc.stderr or '').splitlines() if l.startswith('error')]
        out.append(f'- compile errors: {len(errs)}')
        for l in errs[:10]: out.append(f'  - {l}')
        try:
            import pypdf
            r = pypdf.PdfReader(str(pdf))
            out.append(f'- PDF pages: {len(r.pages)}')
            marks = {'Final experimental results': None, 'Scaling Results': None}
            for i, page in enumerate(r.pages, 1):
                t = page.extract_text() or ''
                for k in marks:
                    if marks[k] is None and k in t: marks[k] = i
            out.append(f"- page where Section 6 (Scaling Results) starts: {marks['Scaling Results']}")
            out.append(f"- page where the AI-DRAFT final section starts: {marks['Final experimental results']}")
            if marks['Scaling Results']:
                out.append(f"- main-content pages before Section 6: {marks['Scaling Results']} "
                           "(3-page minimum applies to Sections 1-5)")
        except Exception as exc:
            out.append(f'- page count skipped: {exc}')
    except Exception as exc:
        out.append(f'- compile skipped: {exc}')
else:
    out.append('- compile skipped: no tectonic binary (set TECTONIC_BIN)')
Path('experiments/final/E13_lint.md').write_text('\n'.join(out)+'\n')
(root/'report/AI_DRAFTS.md').write_text('# AI draft checklist\n\nEvery block below must be rewritten or approved by the authors.\n\n'+
    '\n'.join(f'- report/main.tex lines {a}-{b}' for a,b in drafts)+'\n')
print('\n'.join(out))
