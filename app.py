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
    st.text_area("Compito generato", exercise_text(params), height=520)
    st.download_button("Scarica testo (.txt)", exercise_text(params), "compito_probabilita.txt", "text/plain")

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

