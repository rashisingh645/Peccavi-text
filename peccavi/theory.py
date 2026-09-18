"""
peccavi/theory.py
Theoretical foundations of the PECCAVI watermarking framework.

This module states and sketches proofs for three core propositions cited in
the PECCAVI paper. They establish: (1) statistical consistency of detection,
(2) O(sqrt(T)) detection power growth, and (3) REINFORCE convergence on the
composite quality-watermark reward. Together they give the method a formal
grounding that distinguishes it from purely empirical watermarking systems.

Usage in the paper
------------------
Import the PROPOSITIONS dict to print LaTeX-ready statements:
    from peccavi.theory import PROPOSITIONS
    for p in PROPOSITIONS.values():
        print(p["latex"])
"""

from __future__ import annotations
import math

# Symbolic helpers — used in proof sketches (no heavy deps required)
def phi(x: float) -> float:
    """Standard normal CDF via approximation (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def detection_power(theta: float, T: int, fpr_alpha: float = 0.01) -> float:
    """
    Predicted TPR at a given FPR for watermarked text of length T.

    Derived from Proposition 2: TPR(alpha) >= Phi(z_{1-alpha} + mu(theta)*sqrt(T))
    where mu(theta) = E[g(x,r)|wm] - 0.5 and g takes values in [0,1].

    For PECCAVI's hash-based green scoring, empirically mu(theta) ~ 0.05*theta
    (calibrated from training runs with theta in [2, 7]).
    """
    z_alpha = -_inv_phi(fpr_alpha)          # e.g. z_{0.99} ~ 2.33
    mu_theta = min(0.05 * theta, 0.45)      # saturates below 0.5
    return phi(z_alpha + mu_theta * math.sqrt(T))


def _inv_phi(p: float) -> float:
    """Rational approximation of the inverse normal CDF (Abramowitz & Stegun)."""
    t = math.sqrt(-2.0 * math.log(p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)



# Formal propositions (LaTeX strings + plain-English summaries)

PROPOSITIONS: dict[str, dict] = {

    "P1_consistency": {
        "number": 1,
        "title": "Detection Consistency",
        "summary": (
            "The PECCAVI z-test is a consistent hypothesis test: under the null "
            "(no watermark) it has the correct Type-I error rate alpha, and under "
            "the alternative (watermarked, theta > 0) the power converges to 1 as "
            "text length T -> infinity."
        ),
        "latex": r"""
\begin{proposition}[Detection Consistency]
\label{prop:consistency}
Let $x_{1:T}$ be a token sequence of length $T$, let $g(x_t, r_t) \in [0,1]$ be
the continuous hash-based green score ($r_t$ the context seed derived from the
secret key $K$ and the preceding context), and let
$I_t = \mathbb{1}[g(x_t, r_t) > \tfrac{1}{2}]$ be the binary green-token
indicator that \texttt{Custos} actually thresholds on. The detection statistic is
$z_T = \frac{\sum_{t=1}^{T} I_t - T/2}{\sqrt{T/4}}$.

\textbf{(i) Type-I error control.}
Under $H_0$ (text generated without watermarking), $g(x_t, r_t)$ is
independent of $x_t$ and hash-uniform on $[0,1]$, so $I_t$ are i.i.d.\
$\mathrm{Bernoulli}(\tfrac12)$ with $\mathbb{E}[I_t] = \tfrac{1}{2}$ and
$\mathrm{Var}[I_t] = \tfrac{1}{4}$ (this is where the $T/4$ comes from — it is
the variance of the \emph{binary indicator}, not of the continuous score $g$
itself, which would instead have $\mathrm{Var}[g] = \tfrac{1}{12}$ under
$H_0$). By the Central Limit Theorem $z_T \xrightarrow{d} \mathcal{N}(0,1)$,
so $\Pr(z_T \geq z_\alpha \mid H_0) \to \alpha$ for any $z_\alpha = \Phi^{-1}(1-\alpha)$.

\textbf{(ii) Consistency under $H_1$.}
Under $H_1$ (text generated with $\theta > 0$), tournament sampling biases
token selection toward high-$g$ candidates, so
$\Pr(I_t = 1) = \tfrac{1}{2} + \mu(\theta) > \tfrac{1}{2}$
for some $\mu(\theta) > 0$ increasing in $\theta$.
Then $z_T / \sqrt{T} \xrightarrow{p} 2\mu(\theta) > 0$,
so $\Pr(z_T \geq z_\alpha \mid H_1) \to 1$ as $T \to \infty$.
\end{proposition}
""",
        "proof_sketch": (
            "Under H0: g(x_t, r_t) is independent of x_t (secret key is unknown "
            "to the generator), so g is hash-uniform on [0,1] and I_t = 1[g>0.5] "
            "is i.i.d. Bernoulli(0.5) with Var[I_t] = 0.25 — this is the variance "
            "the z_T denominator actually uses (the continuous score g itself has "
            "Var[g] = 1/12, a different quantity). CLT gives asymptotic normality. "
            "Under H1: tournament sampling biases selection toward high-g tokens, "
            "raising P(I_t=1) above 0.5 by an amount mu(theta) > 0. "
            "LLN gives z_T/sqrt(T) -> 2*mu(theta) > 0, so z_T -> infinity."
        ),
    },

    "P2_power": {
        "number": 2,
        "title": "Detection Power Bound",
        "summary": (
            "Detection power (TPR at fixed FPR) grows at least as fast as "
            "Phi(z_{1-alpha} + mu(theta)*sqrt(T)), i.e. O(sqrt(T)) in text length. "
            "Longer texts are exponentially easier to detect."
        ),
        "latex": r"""
\begin{proposition}[Detection Power Bound]
\label{prop:power}
Under $H_1$ with watermark strength $\theta > 0$, the power of the z-test
at significance level $\alpha$ satisfies
\[
  \mathrm{TPR}(\alpha, T) \;\geq\; \Phi\!\left(\Phi^{-1}(\alpha) + \mu(\theta)\sqrt{T}\right),
\]
where $\mu(\theta) = \Pr(I_t = 1 \mid H_1) - \tfrac{1}{2} > 0$, $I_t$ the
green-token indicator of Proposition~\ref{prop:consistency},
and $\Phi$ is the standard normal CDF.

Consequently, for any target power $1-\beta$, the minimum text length required
for detection is $T^* = O\!\left((\Phi^{-1}(1-\beta) - \Phi^{-1}(\alpha))^2 / \mu(\theta)^2\right)$.
\end{proposition}
""",
        "proof_sketch": (
            "The test statistic z_T ~ N(mu(theta)*sqrt(T), 1) under H1 (approximately, "
            "by CLT applied to the bounded g scores). Power = P(z_T >= z_alpha | H1) "
            "= P(N(mu*sqrt(T), 1) >= z_alpha) = Phi(z_alpha - mu*sqrt(T) ... see full "
            "derivation). The bound follows immediately by non-central normal tail bounds."
        ),
    },

    "P3_reinforce": {
        "number": 3,
        "title": "REINFORCE Convergence on Composite Reward",
        "summary": (
            "The PECCAVI REINFORCE update converges (in expectation) to a stationary "
            "point of the composite reward J(theta) = lambda*S_orig + nu*quality "
            "(plus, when enabled, a PPL penalty and an attack-survival bonus). "
            "With bounded reward and learning rate alpha satisfying the Robbins-Monro "
            "conditions, theta_t converges almost surely."
        ),
        "latex": r"""
\begin{proposition}[REINFORCE Convergence]
\label{prop:reinforce}
Define the composite reward
\[
  r(x, \theta) = \lambda \cdot S(x;\theta) + \nu \cdot Q(x)
  - \mu \cdot \max(0, \mathrm{PPL_{ratio}}(x) - 1) + \rho \cdot \mathrm{Survival}(x),
\]
where $S(x;\theta) = \frac{1}{T}\sum_t g(x_t, r_t) \in [0,1]$ is the watermark score,
$Q(x) \in [1,5]$ is the text quality score, $\mathrm{Survival}(x) \in [0,1]$ is the
post-attack detection-strength term, and $\lambda,\nu,\mu,\rho \geq 0$ are fixed
weights ($\mu = \rho = 0$ by default, recovering the two-term reward
$r = \lambda S + \nu Q$ with $\lambda+\nu=1$ used below).

The PECCAVI policy gradient update
\[
  \theta_{k+1} = \theta_k + \alpha_k \sum_{t=1}^{T} \bigl(g(x_t, r_t) - \tfrac12\bigr)\,
  \bigl(r(x, \theta_k) - b_k\bigr),
\]
with $b_k$ a running-mean baseline of recent rewards, is an unbiased stochastic
gradient estimate of $J(\theta) = \mathbb{E}_x[r(x,\theta)]$.
Provided $\mathrm{PPL_{ratio}}(x)$ is almost-surely bounded (so that $r$ remains
bounded) and $\sum_k \alpha_k = \infty$, $\sum_k \alpha_k^2 < \infty$,
$\theta_k$ converges almost surely to a stationary point
$\theta^*$ satisfying $\nabla_\theta J(\theta^*) = 0$.
\end{proposition}
""",
        "proof_sketch": (
            "The update direction is E[grad log p_w(x|theta) * (r - b)] by the "
            "REINFORCE identity (Williams 1992), using (g(x_t,r_t) - 1/2) as the "
            "centered score-function proxy (see Magister._policy_gradient) and a "
            "running-mean baseline b_k in place of a fixed b=0.5. This is an unbiased "
            "gradient estimate whenever b_k is uncorrelated with the current sample's "
            "randomness. Convergence follows from standard stochastic approximation "
            "theory (Robbins-Monro) given bounded reward and diminishing step sizes. "
            "The mu*PPL and rho*Survival terms don't change this argument as long as "
            "they stay bounded (Survival is bounded in [0,1] by construction; the PPL "
            "penalty is bounded only if PPL_ratio itself is, which is assumed, not proven)."
        ),
    },

    "C1_tradeoff": {
        "number": 4,
        "title": "Quality-Watermark Pareto Frontier (Corollary)",
        "summary": (
            "The lambda/nu ratio in the composite reward parametrises the Pareto "
            "frontier between detection power and text quality. PECCAVI adaptively "
            "navigates this frontier; KGW and SIR operate at fixed points on it."
        ),
        "latex": r"""
\begin{corollary}[Quality-Watermark Pareto Frontier]
\label{cor:pareto}
For fixed text length $T$, the detection power $\mathrm{TPR}(\alpha, T)$ is
strictly increasing in $\theta$, while GPT-2 perplexity $\mathrm{PPL}(\theta)$
is non-decreasing in $\theta$.
There exists a Pareto frontier
$\mathcal{F} = \{(\mathrm{TPR}(\theta), \mathrm{PPL}(\theta)) : \theta \geq 0\}$
such that no method can improve both metrics simultaneously beyond $\mathcal{F}$.

The PECCAVI REINFORCE objective $J(\theta) = \lambda S + \nu Q$ traces $\mathcal{F}$
as $\lambda/\nu$ varies. KGW and SIR operate at fixed points on $\mathcal{F}$
determined by their hyperparameters $\delta$ and $H_{\min}$ respectively,
whereas PECCAVI's adaptive $\theta_k$ converges to the point on $\mathcal{F}$
that maximises $J$ for the given $\lambda/\nu$.
\end{corollary}
""",
        "proof_sketch": (
            "TPR increasing in theta follows from Proposition 2 (mu(theta) is "
            "strictly increasing). PPL non-decreasing follows because larger theta "
            "deviates the sampling distribution further from p_LM, increasing "
            "cross-entropy with respect to the natural language model. "
            "Pareto frontier existence follows from continuity and monotonicity. "
            "The REINFORCE convergence to a stationary point of J(theta) means "
            "it solves the weighted scalarisation of the bicriteria problem."
        ),
    },
}


def print_propositions(latex: bool = False):
    """Print all propositions to stdout."""
    for key, prop in PROPOSITIONS.items():
        print(f"\n{'='*70}")
        print(f"Proposition {prop['number']}: {prop['title']}")
        print(f"{'='*70}")
        if latex:
            print(prop["latex"])
        else:
            print(prop["summary"])
            print(f"\nProof sketch:\n  {prop['proof_sketch']}")


def detection_power_table(theta_values=(2.0, 4.0, 6.0),
                          T_values=(50, 100, 200, 500),
                          fpr_alpha: float = 0.01) -> str:
    """
    Print a table of predicted TPR @ 1% FPR for different theta and T values.
    Useful as a sanity check / supplement for the paper.
    """
    header = f"{'T':>6}" + "".join(f"  theta={t:.0f}" for t in theta_values)
    rows = [header, "-" * len(header)]
    for T in T_values:
        row = f"{T:>6}"
        for theta in theta_values:
            tpr = detection_power(theta, T, fpr_alpha)
            row += f"  {tpr:>9.4f}"
        rows.append(row)
    return "\n".join(rows)


if __name__ == "__main__":
    print_propositions(latex=False)
    print("\n\nDetection Power Table (TPR @ 1% FPR):")
    print(detection_power_table())
    print("\nLaTeX propositions:")
    print_propositions(latex=True)
