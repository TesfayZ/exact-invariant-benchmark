"""Generate the paper's vector plot from cached verdicts; no plotting dependency.
Run from the repository root, then compile the paper with pgfplots installed.
"""
import json
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)


audit = json.load(open('results/policy_audit.json'))
models = [('openai/gpt-oss-120b', 'Groq'), ('gpt-oss-120b', 'Cerebras'),
          ('gemini-3.5-flash-lite', 'Gemini')]
ORDER = [('acrobot', 'Acrobot'), ('acrobot_pert', 'Acrobot\\_p'), ('kepler', 'Kepler'), ('kepler_pert', 'Kepler\\_p'), ('moiety_network', 'Moiety'), ('moiety_network_pert', 'Moiety\\_p'), ('rigid_body_1_2_3', 'Rigid'), ('rigid_body_3_5_11_pert', 'Rigid\\_p'), ('twisted_cubic', 'Twisted'), ('twisted_cubic_pert', 'Twisted\\_p')]
out = [r'\begin{tikzpicture}']
for panel, (model, label) in enumerate(models):
    labels, links, strict, valid = [], [], [], []
    for system, short in ORDER:
        for cond in 'AB':
            y = 20-len(labels)
            labels.append(short+', '+cond)
            c = next(c for c in audit['cells'] if (c['model'],c['system'],c['condition'])==(model,system,cond))
            ref, tv = [100*c[k]/c['n'] for k in ('reference','task_valid')]
            strict.append(f'({ref:g},{y+0.08:g})')
            valid.append(f'({tv:g},{y-0.08:g})')
            links += [f'({ref:g},{y+0.08:g}) ({tv:g},{y-0.08:g})', '(nan,nan)']
    out.append(r'\begin{axis}['+f'at={{({panel*4.5}cm,0)}},anchor=west,'+
        r'width=4.5cm,height=10cm,xmin=-4,xmax=104,ymin=0.4,ymax=20.6,'+
        r'xtick={0,50,100},xlabel={Rate (\%)},ytick={'+','.join(map(str,range(20,0,-1)))+'},'+
        r'yticklabels={'+(','.join('{'+s+'}' for s in labels) if panel==0 else '')+'},'+
        r'tick label style={font=\scriptsize},label style={font=\small},xmajorgrids=true,grid style={gray!15},'+
        r'axis y line*=left,axis x line=bottom,y axis line style={draw=none},ytick style={draw=none},'+
        f'title={{{label}}},'+r'legend style={font=\scriptsize,draw=none,at={(1.5,-0.13)},anchor=north,legend columns=2}]')
    out.append(r'\addplot[gray,thin,unbounded coords=jump] coordinates {'+' '.join(links)+'};')
    out.append(r'\addplot[only marks,mark=o,blue!75!black] coordinates {'+' '.join(strict)+'};')
    out.append(r'\addplot[only marks,mark=square*,mark size=1.5pt,orange!80!black] coordinates {'+' '.join(valid)+'};')
    if panel==0: out.append(r'\legend{,Reference recovery,Task success}')
    out.append(r'\end{axis}')
out.append(r'\end{tikzpicture}')
open('docs/figure_metric_gap.tex','w').write('\n'.join(out)+'\n')
print('wrote complete 60-cell docs/figure_metric_gap.tex')

# Optional standalone export: make -C docs figure
open('docs/figure_metric_gap_standalone.tex', 'w').write(
    r"\documentclass[tikz,border=5pt]{standalone}" + "\n" +
    r"\usepackage{pgfplots}\pgfplotsset{compat=1.18}" + "\n" +
    r"\begin{document}" + "\n" +
    r"\InputIfFileExists{docs/figure_metric_gap.tex}{}{\input{figure_metric_gap.tex}}" + "\n" +
    r"\end{document}" + "\n")
