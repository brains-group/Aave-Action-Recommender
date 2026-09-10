# Clean and marked revision builds

`paper.tex` is the single manuscript source. It defaults to **final** mode for the
`changes` package, so the clean PDF contains accepted text only. `paper-marked.tex`
defines `\ShowChanges` and inputs that same file, selecting **draft** changes markup.
Do not use the document-class `draft` option: that can alter figure rendering.

In Overleaf, select `paper.tex` as the main document for the clean PDF, then select
`paper-marked.tex` for the marked PDF. Both use the same source, figures and bibliography.
For a local TeX installation with all dependencies:

```bash
latexmk -pdf -outdir=build/clean paper.tex
latexmk -pdf -outdir=build/marked paper-marked.tex
```

Alternatively, with Tectonic installed:

```bash
mkdir -p build/clean build/marked
tectonic --keep-logs --outdir build/clean paper.tex
tectonic --keep-logs --outdir build/marked paper-marked.tex
```

The current machine's system pdflatex lacks required packages. Both manuscript builds
were instead checked with a session-local Tectonic engine; its version/hash and build
results are recorded in `revision-verification.json`. No system TeX packages were replaced.
Original compiled paper assets are preserved. Generated PDFs live under `revision-build/`.

## Marking future edits

```tex
\added{New wording.}
\deleted{Removed wording.}
\replaced{New wording.}{Original wording.}
\replaced[comment={R2-2: explanation of the change}]{New wording.}{Original wording.}
```

The order for replacement is **new, then old**. Preserve original wording within the
markup. `commandnameprefix=ifneeded` retains the existing conflict-handling setting;
if a command already exists, changes uses the corresponding `\ch...` command. Consult
the build log before assuming a name such as `\comment` belongs to changes.
Comments are footnotes in the marked version and disappear in final mode.

Do not wrap whole figures, tables, labels or fragile structural commands indiscriminately.
Keep operative structure safe, mark the visible change/comment, preserve the original
asset/source, and add the exact before/after to this ledger. Record figure replacements
by filename and checksum. Git history complements this ledger and the marked PDF.

## Exact change ledger

### R1-P1 — evaluation subsection reference correction

Location: `Evaluation Methodology: Replay-Based Validation`.
Original command, as reported by the author:

```tex
\Cref{sec:experimental_validation}
```

The author had already changed it to the following before this work began:

```tex
\label{sec:experimental_validation}
```

The live label remains outside markup. The marked source now includes:

```tex
\label{sec:experimental_validation}
\deleted[comment={R1-P1: Replaced the accidental \texttt{\string\Cref} command with \texttt{\string\label} for this subsection, fixing the unresolved cross-reference.}]{\mbox{\Cref{sec:experimental_validation}}}
```

This is a structural replacement: a label has no visible text to color as an addition.
The marked PDF strikes out the accidentally rendered reference and explains the new label
in a changes-package comment. The clean PDF omits both. Keeping the live label outside
markup avoids duplicate label writes or fragile effects when the change list is processed.
The original reference now resolves against the corrected label in the marked PDF; we do
not manufacture an unresolved `??` solely to recreate the previous typesetting error.

The untouched baseline copied at the start of this task already contains the author's
correct label. It is not misrepresented as a copy of the pre-correction submitted source.
`source-change.json` records both that baseline and the author-reported original command.

### REV-INFRA-001 — clean/marked package selection

Before: `\usepackage[final,commandnameprefix=ifneeded]{changes}` (with a commented draft
alternative). After: an `\ifdefined\ShowChanges` conditional selects draft or final;
`commentmarkup=footnote` makes correction notes readable. The old commented alternative
is preserved for historical context. A three-line `paper-marked.tex` wrapper enables draft.
This is build infrastructure, not a scientific-text change; the exact source is preserved
in the baseline and patch. Package configuration cannot itself be wrapped in text markup.

### REV-INFRA-002 — existing xcolor option clash

Before: acmart loads xcolor before `\usepackage[table]{xcolor}`, causing an option clash
with the verified TeX bundle. After: `\PassOptionsToPackage{table}{xcolor}` precedes
`\documentclass`; the existing package line is retained. No paper wording or result changes.
This package-order correction is recorded here because preamble commands cannot safely
be shown through body-text changes macros.

Reference: [official changes manual, version 4.2.1](https://mirrors.ibiblio.org/CTAN/macros/latex/contrib/changes/changes.english.pdf).

## Verified local engine

The engine used on this machine is
`/home/spadef/.codex/sessions/2026/09/08/manuscript-revision/tectonic`, with its downloaded
TeX cache at `/home/spadef/.codex/sessions/2026/09/08/manuscript-revision/tex-cache`.
Use that executable in place of `tectonic` above and set `XDG_CACHE_HOME` to that cache
path to reuse the downloaded packages. These session paths are not a portable dependency;
a standard complete TeX installation or Overleaf can compile the same sources.
Both editions have no undefined references/citations. Existing overfull/underfull boxes,
font-substitution and image-description warnings remain for the presentation workstream.
