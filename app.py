import math
from io import BytesIO

import streamlit as st
from scipy.stats import binom, hypergeom, norm, poisson


st.set_page_config(page_title="Compito di probabilità", page_icon="📘", layout="wide")


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def build_exercises(p):
    n_bin, p_bin = p["n_bin"], p["p_bin"]
    bin_answer = binom.sf(2, n_bin, p_bin) / binom.sf(1, n_bin, p_bin)

    N_hyp, n_hyp, good_hyp = p["N_hyp"], p["n_hyp"], p["good_hyp"]
    # Y counts successes in a population with good_hyp successes.
    hyp_den = hypergeom.sf(1, N_hyp, good_hyp, n_hyp)
    hyp_num = hypergeom.pmf(2, N_hyp, good_hyp, n_hyp)
    hyp_answer = hyp_num / hyp_den if hyp_den else float("nan")

    lambda_pois = math.log(2)
    pois_answer = (lambda_pois, math.exp(-lambda_pois) * lambda_pois)

    total, defective = p["total"], p["defective"]
    q = 1 - defective / total
    z_gt_2 = q**2
    delta = q * (1 - q)
    min_n = math.floor(math.log(1 / 3) / math.log(q)) + 1

    mu_h, mu_b, sigma = p["mu_h"], p["mu_b"], p["sigma"]
    alpha = 0.8 * norm.sf(2, mu_h, sigma) + 0.2 * norm.sf(2, mu_b, sigma)
    beta = 0.8 * norm.sf(2, mu_h, sigma) / alpha
    gamma = 0.2 * norm.cdf(2, mu_b, sigma) / (
        0.8 * norm.cdf(2, mu_h, sigma) + 0.2 * norm.cdf(2, mu_b, sigma)
    )

    m, pa, pb, pc = p["m"], p["pa"], p["pb"], p["pc"]
    cov_xy = -m * pa * pb
    rho_xy = -math.sqrt(pa * pb / ((1 - pa) * (1 - pb)))
    cov_x_yz = -m * pa * (pb + pc)

    return {
        "bin_answer": bin_answer,
        "hyp_answer": hyp_answer,
        "pois_answer": pois_answer,
        "z_gt_2": z_gt_2,
        "delta": delta,
        "min_n": min_n,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "cov_xy": cov_xy,
        "rho_xy": rho_xy,
        "cov_x_yz": cov_x_yz,
    }


def exercise_text(p):
    return f"""COMPITO DI CALCOLO DELLE PROBABILITÀ

Nome e cognome: ________________________________

1. Sia X ~ Bin(n={p['n_bin']}, p={p['p_bin']}). Calcolare P(X>2 | X>1).
   Sia Y ipergeometrica con N={p['N_hyp']}, n={p['n_hyp']} e {p['good_hyp']} elementi favorevoli.
   Calcolare P(Y=2 | Y>1).
   Sia T ~ Poisson con P(T=0)=1/2. Calcolare E(T) e P(T=1).

2. Da un lotto di {p['total']} componenti, di cui {p['defective']} difettosi, si effettuano estrazioni con restituzione.
   L'esperimento si interrompe alla prima estrazione di un pezzo difettoso.
   Calcolare P(Z>2), la probabilità di ottenere il primo difettoso alla quinta estrazione
   sapendo che le prime tre sono non difettose, e il minimo n tale che P(nessun difettoso nelle prime n estrazioni) < 1/3.

3. Un messaggio è umano con probabilità 0.8 e bot con probabilità 0.2.
   X|H ~ N({p['mu_h']}, {p['sigma']}²), X|H^c ~ N({p['mu_b']}, {p['sigma']}²).
   Calcolare P(X>2), P(H|X>2) e P(H^c|X<2).

4. Per {p['m']} messaggi, P(spam)={p['pa']}, P(promozione)={p['pb']}, P(ordinario)={p['pc']}.
   Calcolare Cov(X,Y), rho(X,Y) e Cov(X,Y+Z).
"""


def latex_text(p):
    return rf"""\documentclass[12pt]{{exam}}
\usepackage[italian]{{babel}}
\usepackage{{amsmath,amssymb}}
\begin{{document}}

\noindent Cognome e Nome: \dotfill\\
\begin{{center}}
\textbf{{CALCOLO DELLE PROBABILITÀ -- 15 Settembre 2026 -- MAT-AI}}
\end{{center}}

Motivare dettagliatamente le risposte.

\begin{{questions}}

\question Rispondere ai seguenti quesiti.
\begin{{enumerate}}
\item Sia $X\sim Bin(n={p['n_bin']},p={p['p_bin']})$, calcolare
\[P(X>2\mid X>1)=\]
\item Sia $Y\sim H(N={p['N_hyp']},n={p['n_hyp']},K={p['good_hyp']})$,
calcolare $P(Y=2\mid Y>1)$.
\item Sia $T\sim Poisson$ con $P(T=0)=1/2$, calcolare $E(T)$ e $P(T=1)$.
\end{{enumerate}}

\question Da un lotto di ${{{p['total']}}}$ componenti, di cui ${{{p['defective']}}}$ difettosi,
si effettuano estrazioni con restituzione. L'esperimento viene interrotto la prima volta
che viene estratto un pezzo difettoso. Indicando con $Z$ il numero di estrazioni,
calcolare $P(Z>2)$, la probabilità di ottenere per la prima volta un pezzo difettoso
alla quinta estrazione sapendo che le prime tre sono non difettose, e il minimo $n$
affinché la probabilità di non estrarre componenti difettosi nelle prime $n$ estrazioni
sia minore di $1/3$.

\question Un sistema di Intelligenza Artificiale deve stabilire se un messaggio proviene
da un utente umano ($H$) oppure da un bot ($H^c$). Si ha $P(H)=0.8$ e $P(H^c)=0.2$.
Si assume che $X\mid H\sim N({p['mu_h']},{p['sigma']}^2)$ e
$X\mid H^c\sim N({p['mu_b']},{p['sigma']}^2)$. Calcolare $\alpha=P(X>2)$,
$\beta=P(H\mid X>2)$ e $\gamma=P(H^c\mid X<2)$.

\question Dei messaggi vengono classificati in tre categorie: spam, promozioni e ordinari.
Per $m={p['m']}$ messaggi, si assumano probabilità
$P(A_i)={p['pa']}$, $P(B_i)={p['pb']}$, $P(C_i)={p['pc']}$.
Calcolare $\operatorname{{Cov}}(X,Y)$, $\rho_{{X,Y}}$ e
$\operatorname{{Cov}}(X,Y+Z)$.

\end{{questions}}
\end{{document}}
"""


def solution_latex(p, a):
    text = f"""### Soluzioni dettagliate

**1.**
$$
P(X>2\mid X>1)=P(X>2)/P(X>1)={fmt(a['bin_answer'])}
\]
con $P(X=k)=C({p['n_bin']},k){p['p_bin']}^k(1-{p['p_bin']})^({p['n_bin']}-k)$.
\[
P(Y=2\mid Y>1)=P(Y=2)/P(Y>1)={fmt(a['hyp_answer'])}
\]
con $P(Y=k)=C({p['good_hyp']},k)C({p['N_hyp']-p['good_hyp']},{p['n_hyp']}-k)/C({p['N_hyp']},{p['n_hyp']})$.
\[
e^(-\lambda)=1/2\Rightarrow \lambda=\log 2,\quad E(T)=\log 2,\quad P(T=1)=\lambda e^(-\lambda)=(\log 2)/2.
\]

**2.** Ponendo $q=({p['total']}-{p['defective']})/{p['total']}$:
\[
P(Z>2)=q^2={a['z_gt_2']:.6f},\quad \delta=q(1-q)={a['delta']:.6f}.
\]
\[
q^n<1/3\Rightarrow n>\log(1/3)/\log(q),\quad n={a['min_n']}.
\]

**3.** Con $\Phi$ funzione di ripartizione della normale standard:
\[
\alpha=0.8P(X>2\mid H)+0.2P(X>2\mid H^c)={a['alpha']:.6f}.
\]
\[
\beta=P(H\mid X>2)=0.8P(X>2\mid H)/\alpha={a['beta']:.6f}.
\]
\[
\gamma=P(H^c\mid X<2)=0.2P(X<2\mid H^c)/P(X<2)={a['gamma']:.6f}.
\]

**4.** Per la multinomiale:
\[
Cov(X,Y)=-mp_Ap_B={a['cov_xy']:.6f},\quad
rho(X,Y)=-sqrt(p_Ap_B/((1-p_A)(1-p_B)))={a['rho_xy']:.6f}.
\]
Poiché $Y+Z=m-X$:
\[
Cov(X,Y+Z)=Cov(X,m-X)=-Var(X)=-mp_A(1-p_A)={a['cov_x_yz']:.6f}.
\]
"""
    return text.replace("\\[", "$$").replace("\\]", "$$")


st.title("📘 Generatore di compiti di probabilità")
st.caption("Prima versione: modifica i parametri e genera una variante del compito.")

with st.sidebar:
    st.header("Parametri")
    n_bin = st.number_input("Binomiale: n", min_value=1, value=4, step=1)
    p_bin = st.number_input("Binomiale: p", min_value=0.001, max_value=0.999, value=1 / 3, format="%.3f")
    N_hyp = st.number_input("Ipergeometrica: N", min_value=1, value=9, step=1)
    n_hyp = st.number_input("Ipergeometrica: n", min_value=1, max_value=int(N_hyp), value=3, step=1)
    good_hyp = st.number_input("Ipergeometrica: favorevoli", min_value=0, max_value=int(N_hyp), value=6, step=1)
    st.divider()
    total = st.number_input("Lotto: componenti totali", min_value=2, value=10, step=1)
    defective = st.number_input("Lotto: componenti difettosi", min_value=1, max_value=int(total - 1), value=3, step=1)
    st.divider()
    mu_h = st.number_input("Normale: media umano", value=2.0)
    mu_b = st.number_input("Normale: media bot", value=3.0)
    sigma = st.number_input("Normale: deviazione standard", min_value=0.001, value=1.0)
    st.divider()
    m = st.number_input("Multinomiale: numero messaggi", min_value=1, value=4, step=1)
    pa = st.number_input("P(spam)", min_value=0.0, max_value=1.0, value=0.2, format="%.2f")
    pb = st.number_input("P(promozione)", min_value=0.0, max_value=1.0, value=0.3, format="%.2f")
    pc = st.number_input("P(ordinario)", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
    show_solutions = st.checkbox("Mostra soluzioni", value=True)

params = locals()
if abs(pa + pb + pc - 1) > 1e-9:
    st.error("Le probabilità della multinomiale devono avere somma 1.")
    st.stop()

answers = build_exercises(params)
tab_task, tab_answers = st.tabs(["Testo del compito", "Risultati"])
with tab_task:
    st.subheader("Testo del compito")
    st.text_area("Versione leggibile", exercise_text(params), height=420)
    st.download_button("Scarica testo (.txt)", exercise_text(params), "compito_probabilita.txt", "text/plain")
    st.subheader("Sorgente LaTeX")
    st.code(latex_text(params), language="latex")
    st.download_button("Scarica sorgente LaTeX (.tex)", latex_text(params), "compito_probabilita.tex", "text/plain")

with tab_answers:
    if not show_solutions:
        st.info("Le soluzioni sono nascoste. Attiva 'Mostra soluzioni' nella barra laterale.")
    else:
        st.subheader("Risultati numerici")
        st.write(f"1. Binomiale: **{fmt(answers['bin_answer'])}**")
        st.write(f"1. Ipergeometrica: **{fmt(answers['hyp_answer'])}**")
        st.write(f"1. Poisson: E(T) = **{fmt(answers['pois_answer'][0])}**, P(T=1) = **{fmt(answers['pois_answer'][1])}**")
        st.write(f"2. P(Z>2) = **{fmt(answers['z_gt_2'])}**, δ = **{fmt(answers['delta'])}**, n minimo = **{answers['min_n']}**")
        st.write(f"3. α = **{fmt(answers['alpha'])}**, β = **{fmt(answers['beta'])}**, γ = **{fmt(answers['gamma'])}**")
        st.write(f"4. Cov(X,Y) = **{fmt(answers['cov_xy'])}**, ρ = **{fmt(answers['rho_xy'])}**, Cov(X,Y+Z) = **{fmt(answers['cov_x_yz'])}**")
        st.subheader("Soluzioni con formule")
        st.markdown(solution_latex(params, answers))

