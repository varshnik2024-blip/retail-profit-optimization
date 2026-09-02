"""Rough page-budget estimate for the IEEE paper (no LaTeX engine needed)."""
import re

t = open("main.tex", encoding="utf-8").read()
body = t[t.index(r"\begin{abstract}"):t.index(r"\begin{thebibliography}")]
bib = t[t.index(r"\begin{thebibliography}"):t.index(r"\balance")]


def words(s):
    s = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " EQ ", s, flags=re.S)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}$\\&_^%~\[\]]", " ", s)
    return len([w for w in s.split() if any(c.isalpha() for c in w)])


bw, rw = words(body), words(bib)
print(f"body words           : {bw:>5}")
print(f"bibliography entries : {t.count(chr(92) + 'bibitem'):>5}")
print(f"equations            : {t.count('begin{equation}'):>5}")
print(f"double-col figures   : {t.count('begin{figure*}'):>5}")
print(f"tables               : {t.count('begin{table}'):>5}")

fig_h = [3.4, 2.5, 2.4, 2.3, 2.4]          # figure heights, inches
per_page = 2 * 9.6                          # column-inches available per page
figc = sum(2 * (h + 0.55) for h in fig_h) / per_page
tblc = 3 * 2.2 / per_page
txtc = bw / 1000.0                          # ~1000 words/page of 2-col body
refc = rw / 1700.0                          # references set at 8pt

print("\nSPACE ESTIMATE (IEEEtran conference, A4)")
print(f"  title + abstract  ~ 0.30")
print(f"  body text         ~ {txtc:.2f}")
print(f"  5 wide figures    ~ {figc:.2f}")
print(f"  3 tables          ~ {tblc:.2f}")
print(f"  references        ~ {refc:.2f}")
print("  " + "-" * 27)
total = 0.30 + txtc + figc + tblc + refc
print(f"  TOTAL             ~ {total:.2f} pages")
if total > 6.15:
    print(f"\n  OVER by ~{total - 6.0:.2f} pages -> trim text or shrink figures")
elif total < 5.6:
    print(f"\n  UNDER by ~{6.0 - total:.2f} pages -> figures can be enlarged")
else:
    print("\n  Within tolerance of a 6-page target.")
