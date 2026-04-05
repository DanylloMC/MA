import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Garante exibição correta de caracteres Unicode no terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({
    'font.size': 12,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'figure.figsize': (14, 8)
})

# =============================================================================
# BLOCO 1 — PARÂMETROS DA AERONAVE (UAV ~14 kg)
#
# PREMISSAS:
#   - Eixo de corpo (body axis): x aponta para frente, z aponta para baixo
#   - Simetria bilateral → movimento puramente longitudinal (v=p=r=φ=0)
#   - Empuxo T alinhado com eixo x do corpo (sem componente em z)
#   - Voo de referência nivelado: gammae = 0 → thetae = alphae
#   - Derivadas de estabilidade adimensionais obtidas de ensaios / cálculo
#     e referenciadas às convenções de Cook (Flight Dynamics Principles, Cap. 4)
#   - Densidade do ar: ISA ao nível do mar (rho = 1.225 kg/m3)
#
# NOTA SOBRE NOMENCLATURA (Cook, Cap. 4):
#   CX, CZ : coeficientes de força axial e normal em eixo de corpo
#   Cm     : coeficiente de momento de arfagem (positivo nose-up)
#   Sufixo _u, _w, _q, _wdot, _de : derivadas parciais adimensionais
# =============================================================================

# --- Parâmetros físicos ---
m   = 14.000   # Massa total [kg]
Iy  = 0.385    # Momento de inércia em arfagem [kg.m2]
g   = 9.81     # Aceleração gravitacional [m/s2]
S   = 1.213    # Área da asa [m2]
c   = 0.5048   # Corda aerodinâmica média (c̄) [m]

# --- Condição de voo de referência (trim) ---
# CORREÇÃO: rho = np.rho não existe em NumPy; valor ISA ao nível do mar.
rho     = 1.225              # Densidade do ar [kg/m3]
V0      = 12.0               # Velocidade de trim [m/s]
theta_e = np.deg2rad(9.4)    # Ângulo de arfagem de trim [rad]
alpha_e = theta_e            # gammae = 0  →  thetae = alphae

Ue = V0 * np.cos(alpha_e)    # Componente axial de trim [m/s]
We = V0 * np.sin(alpha_e)    # Componente normal de trim [m/s]
q_dyn_e = 0.5 * rho * V0**2  # Pressão dinâmica de trim [Pa]

# --- Derivadas adimensionais de estabilidade longitudinal ---
# Convenção Cook / Bryan: perturbações normalizadas por V₀ (eixo de corpo).
# Os termos "_wdot" usam normalização adicional por c/(2V₀) (ver Bloco 4).
CX_u     =  0.0076
CX_w     =  0.0483
CX_wdot  =  0.0       # geralmente negligenciável; incluído por completeza
CX_q     =  0.0
CX_de    =  0.0618    # deltae = elevator

CZ_u     = -0.7273
CZ_w     = -3.1245
CZ_wdot  = -0.3997    # efeito de aceleração normal (Madelung)
CZ_q     = -1.2109
CZ_de    = -0.3741

Cm_u     =  0.0340
Cm_w     = -0.2169
Cm_wdot  = -0.5910    # amortecimento por aceleração normal (Madelung)
Cm_q     = -1.2732
Cm_de    = -0.5581

# --- Polar de arrasto e condição de trim ---
CD0  = 0.0806    # Coeficiente de arrasto parasita
k    = 0.0706    # Fator de arrasto induzido  k = 1/(π.AR.e)
CL_alpha = -CZ_w   # Derivada CLα ≈ -CZ_w [1/rad] (eixo de estabilidade ≈ corpo p/ αe pequeno)

# CL de equilíbrio a partir do equilíbrio de força normal:
#   Z_aero_trim = -m·g·cos(θe)   →   CL_e·cos(αe) + CD_e·sin(αe) = m·g·cos(αe)/(q_e·S)
# Para αe pequeno: CL_e ≈ m·g/(q_e·S); a correção de arrasto é < 2% para αe < 15°.
CL_e = m * g * np.cos(alpha_e) / (q_dyn_e * S)   # eq. força normal (exata para CL)
CD_e = CD0 + k * CL_e**2                          # polar de arrasto no trim

# --- Forças de base no trim (eixo de corpo) ---
# No eixo de corpo, projetando as forças de estabilidade (X_s=-D, Z_s=-L) por ângulo αe:
#   X_body = -D·cos(α) + L·sin(α)   (componente axial)
#   Z_body = -D·sin(α) - L·cos(α)   (componente normal, z↓)
#
# CZ_trim: diretamente da eq. de equilíbrio Ẇ=0 →  Z_aero = -m·g·cos(θe)
CZ_trim = -m * g * np.cos(alpha_e) / (q_dyn_e * S)

# CX_trim: da projeção de L e D no eixo x do corpo
CX_trim = (-CD_e * np.cos(alpha_e) + CL_e * np.sin(alpha_e))

# Empuxo de trim: da eq. U̇=0 →  T_trim = m·g·sin(θe) - q_e·S·CX_trim
#   Para αe pequeno: T_trim ≈ D_e (arrasto de trim)
T_trim_analytical = m * g * np.sin(alpha_e) - q_dyn_e * S * CX_trim

print("=" * 62)
print("  PARÂMETROS DA AERONAVE — UAV ~14 kg")
print("=" * 62)
print(f"  Massa                : {m:.1f} kg")
print(f"  Inércia em arfagem   : {Iy:.3f} kg.m2")
print(f"  Área da asa          : {S:.3f} m2")
print(f"  Corda média          : {c:.4f} m")
print(f"  Velocidade de trim   : {V0:.1f} m/s")
print(f"  Densidade do ar (rho)  : {rho:.3f} kg/m3")
print(f"  Pressão dinâmica     : {q_dyn_e:.2f} Pa")
print(f"  Ângulo de trim thetae=alphae : {np.rad2deg(alpha_e):.2f} deg")
print(f"  Ue / We              : {Ue:.3f} / {We:.3f} m/s")
print(f"  CL de trim           : {CL_e:.4f}")
print(f"  CD de trim           : {CD_e:.4f}")
print("=" * 62)

# =============================================================================
# BLOCO 2 — MODELO AERODINÂMICO E EQUAÇÕES DE MOVIMENTO NÃO-LINEARES
#
# PREMISSAS:
#   - EOM em eixo de corpo (Bryan, 3-DOF longitudinal), conforme Cook Cap. 4.
#   - Forças aerodinâmicas descritas por coeficientes adimensionais expressos
#     em perturbações lineares em torno do trim.
#   - Para o modelo NÃO-LINEAR são usados dois termos:
#       (a) coeficientes de base no trim (CX_e, CZ_e, Cm_e = 0 por definição
#           do trim, mas sustentação e arrasto existem como forças brutas)
#       (b) perturbações Δu = U-Ue, Δw = W-We em torno do trim
#   - A pressão dinâmica é calculada com V instântaneo para capturar variações
#     de velocidade nas simulações não-lineares.
#   - Os termos de aceleração (ẇ) NÃO entram no modelo NÃO-LINEAR diretamente
#     (são implícitos nas equações diferenciais); aparecem apenas no modelo
#     linear como correção Madelung (Bloco 4).
#
# EOM (Bryan, p=r=v=0):
#   m(U. + qW) = X_aero + T  − m.g.sin(theta)    [força axial]
#   m(Ẇ − qU) = Z_aero      + m.g.cos(theta)    [força normal — z↓]
#   Iy.q.      = M_aero                       [momento arfagem]
#   theta.         = q                            [cinemática]
# =============================================================================

def aero_forces_moments(U, W, q, delta_e, rho_loc=rho):
    """
    Forças e momento aerodinâmicos COMPLETOS em eixos de corpo.

    Modelo = forças de BASE no trim + perturbações lineares em torno do trim.

    Forças de base (CX_trim, CZ_trim): computadas a partir das condições de
    equilíbrio em Bloco 1; garantem que no ponto de trim exato as EOM se
    satisfazem com T = T_trim_analytical e deltae = 0.

    Perturbações: derivadas adimensionais de estabilidade normalizadas por V0.
      deltae aqui é a PERTURBAÇÃO de deflexão em relação ao elevator de trim.

    Retorna: X_aero [N], Z_aero [N], M_aero [N.m]
    """
    V     = max(np.sqrt(U**2 + W**2), 1e-6)
    q_hat = q * c / (2.0 * V)
    q_dyn = 0.5 * rho_loc * V**2

    delta_u = U - Ue
    delta_w = W - We

    # Coeficiente total = base no trim + perturbação linear
    CX = (CX_trim +
          CX_u * delta_u / V0 +
          CX_w * delta_w / V0 +
          CX_q * q_hat +
          CX_de * delta_e)

    CZ = (CZ_trim +
          CZ_u * delta_u / V0 +
          CZ_w * delta_w / V0 +
          CZ_q * q_hat +
          CZ_de * delta_e)

    # Cm_trim = 0 por definição do estado de referência
    Cm = (Cm_u * delta_u / V0 +
          Cm_w * delta_w / V0 +
          Cm_q * q_hat +
          Cm_de * delta_e)

    X_aero = q_dyn * S * CX
    Z_aero = q_dyn * S * CZ
    M_aero = q_dyn * S * c * Cm

    return X_aero, Z_aero, M_aero


def longitudinal_nonlinear(t, state, delta_e, T, rho_loc=rho):
    """
    EOM longitudinais NÃO-LINEARES em eixo de corpo (Cook / Bryan).

    Estado: x = [U, W, q, theta]
      U : velocidade axial [m/s]
      W : velocidade normal (positiva para baixo no eixo de corpo) [m/s]
      q : taxa de arfagem [rad/s]
      theta : ângulo de arfagem [rad]

    Entradas:
      delta_e : deflexão do elevator [rad]
      T       : empuxo [N] (alinhado com eixo x do corpo)
    """
    U, W, q, theta = state
    X_aero, Z_aero, M_aero = aero_forces_moments(U, W, q, delta_e, rho_loc)

    U_dot     = -q * W - g * np.sin(theta) + (X_aero + T) / m
    W_dot     =  q * U + g * np.cos(theta) + Z_aero / m
    q_dot     = M_aero / Iy
    theta_dot = q

    return [U_dot, W_dot, q_dot, theta_dot]

# =============================================================================
# BLOCO 3 — PONTO DE EQUILÍBRIO (TRIM)
#
# Condição de trim (ṡtate = 0):
#   q = 0,  U. = 0,  Ẇ = 0,  q. = 0
#   theta_trim = alpha_trim + gamma_trim  (com gamma_trim = 0 → theta = alpha)
#
# Variáveis de otimização: [alpha, deltae, T]
# Método: fsolve (Newton-Raphson generalizado)
# =============================================================================

# =============================================================================
# TRIM: calculado ANALITICAMENTE a partir das condições de equilíbrio.
#
# No modelo perturbativo de Cook, o estado de trim [Ue, We, 0, θe] com
# deltae_pert = 0 É por definição o ponto de referência. As forças de base
# (CX_trim, CZ_trim) garantem que as EOM se satisfazem exatamente com T_trim.
#
# Não há necessidade de fsolve: o trim é explicitamente definido pelo modelo.
# Usamos fsolve apenas para VERIFICAÇÃO numérica (deve convergir trivialmente).
# =============================================================================

# Trim analítico
alpha_trim   = alpha_e
delta_e_trim = 0.0          # perturbação nula = elevator na posição de referência
T_trim       = T_trim_analytical
U_trim       = Ue
W_trim       = We
theta_trim   = alpha_e      # γ = 0

print("\n" + "=" * 62)
print("  BLOCO 3 — PONTO DE EQUILÍBRIO (TRIM)")
print("=" * 62)
print(f"\n  Trim analítico (αe dado, equilíbrio de forças):")
print(f"    α  (ângulo de ataque)  : {np.rad2deg(alpha_trim):+.4f} deg")
print(f"    δe (perturbação elev.) : {np.rad2deg(delta_e_trim):+.4f} deg  [= ref. = 0]")
print(f"    T  (empuxo)            : {T_trim:+.3f} N   (≈ CD·q_e·S/cos(α) = {CD_e*q_dyn_e*S/np.cos(alpha_e):.3f} N)")
print(f"    CX_trim                : {CX_trim:.5f}")
print(f"    CZ_trim                : {CZ_trim:.5f}")
print(f"    U_trim / W_trim        : {U_trim:.4f} / {W_trim:.4f} m/s")
print(f"    CL_trim / CD_trim      : {CL_e:.4f} / {CD_e:.4f}")

# Verificação numérica dos resíduos
residuo = longitudinal_nonlinear(0, [U_trim, W_trim, 0.0, theta_trim],
                                 delta_e_trim, T_trim)
print(f"\n  Verificação de resíduos (devem ser ≈ 0):")
print(f"    U̇  : {residuo[0]:.2e} m/s2")
print(f"    Ẇ  : {residuo[1]:.2e} m/s2")
print(f"    q̇  : {residuo[2]:.2e} rad/s2")

tol_residuo = 1e-4
ok = all(abs(r) < tol_residuo for r in residuo[:3])
print(f"\n  Trim válido (|res| < {tol_residuo:.0e}): {'SIM OK' if ok else 'NÃO FAIL'}")
print("=" * 62)

# =============================================================================
# BLOCO 4 — LINEARIZAÇÃO: MATRIZES A E B (Cook, Cap. 5)
#
# Aplicando teoria de pequenas perturbações às EOM de Bryan em eixo de corpo,
# em torno do ponto de trim (U_trim, W_trim, 0, theta_trim).
#
# Estado:  x = [u, w, q, theta]   (perturbações)
# Entrada: delta = [deltae, ΔT]
#
# DERIVADAS DIMENSIONAIS (força/velocidade ou momento/velocidade):
#   Xi = q_e.S.CX_i / (m.V₀)          [i = u, w]
#   Xq = q_e.S.CX_q.c / (2m.V₀)
#   Xde= q_e.S.CX_de / m
#   Zi = q_e.S.CZ_i / (m.V₀)
#   Zq = q_e.S.CZ_q.c / (2m.V₀)
#   Zẇ = q_e.S.CZ_wdot.c / (2m.V₀)   [aceleração normal]
#   Mi = q_e.S.c.Cm_i / (Iy.V₀)
#   Mq = q_e.S.c2.Cm_q / (2.Iy.V₀)
#   Mẇ = q_e.S.c2.Cm_wdot / (2.Iy.V₀)
#
# CORREÇÃO MADELUNG (termos de aceleração ẇ):
#   A equação Z tem o termo Zẇ.ẇ; isolando ẇ:
#     (1 - Zẇ).ẇ = Zu.u + Zw.w + (Zq + Ue).q - g.sin(thetae).theta + Zde.deltae
#   A equação M incorpora Mẇ.ẇ substituindo a expressão acima:
#     Mw* = Mw + Mẇ.Zw / (1 - Zẇ)
#     Mu* = Mu + Mẇ.Zu / (1 - Zẇ)
#     Mq* = Mq + Mẇ.(Zq + Ue) / (1 - Zẇ)
#     Mde*= Mde + Mẇ.Zde / (1 - Zẇ)
#
# MATRIZ A (4×4):
#   ẋ = A.x  (modo livre, deltae = 0)
#
# MATRIZ B (4×2):
#   ẋ = A.x + B.[deltae, ΔT]
# =============================================================================

# --- Derivadas dimensionais de força (normalizadas por m) ---
Xu  = q_dyn_e * S * CX_u    / (m * V0)
Xw  = q_dyn_e * S * CX_w    / (m * V0)
Xq  = q_dyn_e * S * CX_q * c / (2 * m * V0)
Xde = q_dyn_e * S * CX_de   / m
XdT = 1.0 / m                                 # variação de empuxo

Zu   = q_dyn_e * S * CZ_u    / (m * V0)
Zw   = q_dyn_e * S * CZ_w    / (m * V0)
Zq   = q_dyn_e * S * CZ_q * c / (2 * m * V0)
Zwdot= q_dyn_e * S * CZ_wdot * c / (2 * m * V0)
Zde  = q_dyn_e * S * CZ_de   / m

# --- Derivadas dimensionais de momento (normalizadas por Iy) ---
Mu   = q_dyn_e * S * c  * Cm_u    / (Iy * V0)
Mw   = q_dyn_e * S * c  * Cm_w    / (Iy * V0)
Mq   = q_dyn_e * S * c**2 * Cm_q  / (2 * Iy * V0)
Mwdot= q_dyn_e * S * c**2 * Cm_wdot / (2 * Iy * V0)
Mde  = q_dyn_e * S * c  * Cm_de   / Iy

# --- Correção Madelung ---
denom = 1.0 - Zwdot          # (1 - Zẇ)

Zu_  = Zu  / denom
Zw_  = Zw  / denom
Zq_  = (Zq + U_trim) / denom
Zg_  = -g * np.sin(theta_trim) / denom   # contribuição de gravidade na linha w
Zde_ = Zde / denom

Mu_  = Mu  + Mwdot * Zu  / denom
Mw_  = Mw  + Mwdot * Zw  / denom
Mq_  = Mq  + Mwdot * (Zq + U_trim) / denom
Mg_  = Mwdot * (-g * np.sin(theta_trim)) / denom
Mde_ = Mde + Mwdot * Zde / denom

# --- Matriz A (4×4) — estado [u, w, q, theta] ---
#   Linha u:  m.u. = Xu.u + Xw.w + (Xq - We).q - g.cos(thetae).theta
#   Linha w:  (1-Zẇ).ẇ = Zu.u + Zw.w + (Zq+Ue).q - g.sin(thetae).theta
#   Linha q:  q.  = Mu*.u + Mw*.w + Mq*.q + (Mg - g.cos(thetae).Mẇ/denom?).theta
#   Linha theta:  theta.  = q
#
# NOTA: gravidade na linha u  → -g.cos(thetae) (perturbação em theta afeta componente axial)
#       gravidade na linha w  → -g.sin(thetae) (perturbação em theta afeta componente normal)

A = np.array([
    [Xu,   Xw,   Xq - W_trim,          -g * np.cos(theta_trim)],
    [Zu_,  Zw_,  Zq_,                   Zg_                  ],
    [Mu_,  Mw_,  Mq_,                   Mg_                  ],
    [0.0,  0.0,  1.0,                   0.0                  ]
])

# --- Matriz B (4×2) — entradas [deltae, ΔT] ---
B = np.array([
    [Xde,  XdT],
    [Zde_, 0.0],
    [Mde_, 0.0],
    [0.0,  0.0]
])

C = np.eye(4)           # saída = estado completo
D = np.zeros((4, 2))

sys_lin = signal.StateSpace(A, B, C, D)

print("\n" + "=" * 62)
print("  BLOCO 4 — LINEARIZAÇÃO (Cook, Cap. 5)")
print("=" * 62)
print("\n  Derivadas dimensionais de força [1/s]:")
print(f"    Xu={Xu:.4f}  Xw={Xw:.4f}  Xq={Xq:.4f}  Xde={Xde:.4f}")
print(f"    Zu={Zu:.4f}  Zw={Zw:.4f}  Zq={Zq:.4f}  Zwdot={Zwdot:.4f}  Zde={Zde:.4f}")
print(f"    Mu={Mu:.4f}  Mw={Mw:.4f}  Mq={Mq:.4f}  Mwdot={Mwdot:.4f}  Mde={Mde:.4f}")
print(f"\n  Correção Madelung: denominador (1 - Zẇ) = {denom:.4f}")
print(f"\n  Matriz A:\n{np.array2string(A, precision=4, suppress_small=True)}")
print(f"\n  Matriz B:\n{np.array2string(B, precision=4, suppress_small=True)}")
print("=" * 62)

# =============================================================================
# BLOCO 5 — ANÁLISE DE AUTOVALORES: POLOS E MODOS (Cook, Cap. 6)
#
# A equação característica Δ(s) = det(sI - A) = 0 é um polinômio de 4ª ordem.
# Para uma aeronave convencional ela se fatoriza aproximadamente em dois pares
# de raízes complexas conjugadas:
#
#   Δ(s) ≈ (s2 + 2zeta_sp.omegan_sp.s + omegan_sp2) . (s2 + 2zeta_ph.omegan_ph.s + omegan_ph2)
#
# Período Curto (Short-Period — SP):
#   Alta frequência, bem amortecido; dominado por w e q.
#   Aproximação clássica: omegan_sp2 ≈ Zalpha.Malpha/(m.Iy) com Zalpha = -Zw.V0, Malpha = Mw.V0
#
# Fugóide (Phugoid — PH):
#   Baixa frequência, pouco amortecido; dominado por u e theta.
#   Aproximação de Lanchester: omegan_ph ≈ gsqrt2 / V₀
# =============================================================================

eigenvalues, eigenvectors = np.linalg.eig(A)

# Ordena por frequência (|λ|) decrescente: SP vem primeiro
idx = np.argsort(-np.abs(eigenvalues))
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

def modo_info(lam, nome):
    """
    Extrai e imprime os parâmetros modais de um autovalor.
    Trata corretamente modos reais (overdamped) e complexos (underdamped/instável).
    """
    omega_n = abs(lam)
    sigma   = lam.real
    omega_d = abs(lam.imag)
    oscilatorio = omega_d > 1e-6

    if omega_n > 1e-12:
        zeta = -sigma / omega_n
    else:
        zeta = 1.0

    T_period = 2 * np.pi / omega_d if oscilatorio else float('inf')

    # Constante de tempo: τ = 1/|sigma|; para instável = "tempo para dobrar amplitude"
    if sigma < -1e-10:
        tau_label = "Constante τ (amort.)"
        tau_val   = 1.0 / abs(sigma)
    elif sigma > 1e-10:
        tau_label = "T_dobra (instável)"
        tau_val   = np.log(2) / sigma
    else:
        tau_label = "Constante τ"
        tau_val   = float('inf')

    estavel = sigma < 0

    print(f"\n  [{nome}]")
    if oscilatorio:
        print(f"    Tipo         : Oscilatorio  (par complexo conjugado)")
        print(f"    lambda       : {sigma:+.4f} ± {omega_d:.4f}j  rad/s")
    else:
        print(f"    Tipo         : Real (overdamped / aperiodico)")
        print(f"    lambda       : {sigma:+.4f}  rad/s")
    print(f"    omegan       : {omega_n:.4f} rad/s")
    print(f"    zeta         : {zeta:+.4f}  {'(sobremorto: zeta >= 1)' if zeta >= 1.0 and not oscilatorio else ''}")
    print(f"    omegad       : {omega_d:.4f} rad/s")
    print(f"    Periodo      : {T_period:.3f} s" if oscilatorio else f"    Periodo      : N/A (nao-oscilatorio)")
    print(f"    {tau_label:22s}: {tau_val:.3f} s")
    print(f"    Estavel      : {'SIM OK' if estavel else 'NAO FAIL'}")
    return omega_n, zeta, T_period, tau_val


print("\n" + "=" * 62)
print("  BLOCO 5 — AUTOVALORES E MODOS (Cook, Cap. 6)")
print("=" * 62)

# Classificação dos autovalores por |λ| decrescente.
# NOTA: se os dois maiores |λ| forem reais (SP overdamped), o modo SP
# não tem frequência natural clássica — apenas duas constantes de tempo.
print(f"\n  Todos os autovalores (ordenados por |lambda| decrescente):")
for i, lam in enumerate(eigenvalues):
    tipo = "real" if abs(lam.imag) < 1e-6 else "complexo"
    print(f"    [{i}] {lam.real:+.5f} {'+' if lam.imag >= 0 else '-'} {abs(lam.imag):.5f}j   ({tipo})")

# SP: par (ou par real) de maior |λ|
lam_sp = eigenvalues[0]   # maior magnitude
lam_sp2 = eigenvalues[1]  # segundo (pode ser conjugado de lam_sp, ou segundo polo real)
# PH: par complexo de menor |λ|
lam_ph = eigenvalues[2]

wn_sp, zeta_sp, T_sp, tau_sp = modo_info(lam_sp, "PERIODO CURTO — polo 1 (Short-Period)")

# Se SP for overdamped (dois polos reais), mostrar segundo polo também
if abs(lam_sp.imag) < 1e-6 and abs(lam_sp2.imag) < 1e-6:
    _, _, _, tau_sp2 = modo_info(lam_sp2, "PERIODO CURTO — polo 2 (Short-Period)")
    print(f"\n  NOTA: SP overdamped (zeta >= 1). Produto Zalpha*Malpha < 0,")
    print(f"        indicando que a aprox. classica nao se aplica (ver analise de sensibilidade).")
    print(f"        Os dois polos reais SP sao: lambda1={lam_sp.real:.3f} s-1, lambda2={lam_sp2.real:.3f} s-1")

wn_ph, zeta_ph, T_ph, tau_ph = modo_info(lam_ph, "FUGOIDE (Phugoid)")

# --- Verificações analíticas clássicas ---
Zalpha = -Zw * V0       # derivada Zalpha = Z_alpha [m/s²/rad]  (positivo → estável em Z)
Malpha =  Mw * V0       # derivada Malpha = M_alpha [rad/s²/rad] (negativo → estável em M)

# SP: ωn_sp² ≈ Zα·Mα (aprox. de 2ª ordem; só válida quando Mα < 0 e Zα > 0)
ZM_product = Zalpha * Malpha
if ZM_product > 0:
    wn_sp_approx = np.sqrt(ZM_product)
    sp_note = f"sqrt({ZM_product:.1f}) = {wn_sp_approx:.4f} rad/s"
else:
    wn_sp_approx = float('nan')
    sp_note = f"Zalpha*Malpha = {ZM_product:.1f} < 0 → aprox. invalida (SP overdamped)"

# PH: Lanchester → ωn_ph ≈ g√2/V0, ζ_ph ≈ CD/(√2·CL)
wn_ph_approx  = g * np.sqrt(2) / V0
zeta_ph_approx = CD_e / (np.sqrt(2) * CL_e)

print(f"\n  Verificacoes analiticas classicas (Cook, Cap. 6):")
print(f"    SP — Zalpha={Zalpha:.3f}, Malpha={Malpha:.3f}")
print(f"    omegan_sp (aprox.)  : {sp_note}")
print(f"    PH — Lanchester:")
print(f"    omegan_ph (aprox.)  : g*sqrt(2)/V0 = {wn_ph_approx:.4f} rad/s  (calc.: {wn_ph:.4f})")
print(f"    zeta_ph   (aprox.)  : CD/(sqrt(2)*CL) = {zeta_ph_approx:.4f}  (calc.: {zeta_ph:+.4f})")
if zeta_ph < 0:
    print(f"\n  ATENCAO: Fugoide INSTAVEL (zeta < 0, polo na metade direita do plano s).")
    print(f"           Causa provavel: Cm_u = {Cm_u:.4f} > 0 (instabilidade de velocidade).")
    print(f"           A aprox. de Lanchester assume Cm_u = 0 e nao prevê essa instabilidade.")
print("=" * 62)

# --- Figura 1: Mapa de Polos (dois painéis) ---
# Painel esquerdo : visão geral com todos os polos
# Painel direito  : zoom na região de baixa frequência (fugóide + SP-2)
# Isso evita que o polo SP-1 (muito à esquerda) comprima a escala e
# torne os polos do fugóide invisíveis.

def _plot_poles_on_ax(ax, eigs, zoom=False):
    """Plota polos, eixos, rótulos e arcos de amortecimento num eixo."""
    cores = {'SP1': 'tab:blue', 'SP2': 'tab:cyan', 'PH': 'tab:red'}

    # Agrupa polos por modo
    sp1 = eigenvalues[0]
    sp2 = eigenvalues[1]
    ph_p = eigenvalues[2]   # positivo Im
    ph_m = eigenvalues[3]   # negativo Im (conjugado)

    labels = [
        (sp1, f'SP-1\n$\\lambda$={sp1.real:.2f}', cores['SP1']),
        (sp2, f'SP-2\n$\\lambda$={sp2.real:.2f}', cores['SP2']),
        (ph_p, f'Fugóide\n$\\lambda$={ph_p.real:+.3f}±{abs(ph_p.imag):.3f}j', cores['PH']),
        (ph_m, None, cores['PH']),   # conjugado — sem label
    ]

    for lam, lbl, cor in labels:
        marker = 'x' if lam.real < 0 else 'o'   # × estável, ○ instável
        ms = 12 if not zoom else 14
        ax.plot(lam.real, lam.imag, marker=marker, color=cor,
                markersize=ms, markeredgewidth=2.5, zorder=5)
        if lbl and (not zoom or abs(lam.real) < 5):
            # No zoom, anota apenas polos visíveis (fugóide + SP-2)
            offset_x = 0.06 if lam.imag >= 0 else 0.06
            offset_y = 0.08 if lam.imag >= 0 else -0.12
            ax.annotate(lbl,
                        xy=(lam.real, lam.imag),
                        xytext=(lam.real + offset_x, lam.imag + offset_y),
                        fontsize=9, color=cor,
                        arrowprops=dict(arrowstyle='->', color=cor, lw=1.2),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # Linha vertical em σ=0 (margem de estabilidade)
    ax.axvline(0, color='k', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.35, linestyle=':')
    ax.set_xlabel('Re(λ)  [rad/s]', fontsize=11)
    ax.set_ylabel('Im(λ)  [rad/s]', fontsize=11)

    # Arcos de amortecimento constante (linhas radiais a partir da origem)
    r_max = max(abs(eig) for eig in eigs) * 1.15
    for zc in [0.3, 0.5, 0.7, 1.0]:
        angle = np.arccos(zc)        # ângulo do eixo negativo real
        for sign in [+1, -1]:
            r_arr = np.linspace(0, r_max, 200)
            ax.plot(-r_arr * zc, sign * r_arr * np.sin(angle),
                    color='gray', linewidth=0.7, linestyle='--', alpha=0.45)
        if not zoom:
            # Rótulo do arco
            rx = -r_max * zc * 0.85
            ry =  r_max * np.sin(angle) * 0.85
            ax.text(rx, ry, f'ζ={zc}', fontsize=7, color='gray', ha='center',
                    rotation=np.rad2deg(-np.arctan2(np.sin(angle), zc)))


fig1, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle('Mapa de Polos — Dinâmica Longitudinal (Cook, Cap. 6)', fontsize=13)

# --- Painel esquerdo: todos os polos ---
_plot_poles_on_ax(ax_full, eigenvalues, zoom=False)
# Escala automática com margem
r_all = max(abs(l) for l in eigenvalues)
ax_full.set_xlim(-r_all * 1.25, r_all * 0.25)
ax_full.set_ylim(-r_all * 0.15, r_all * 0.15)
ax_full.set_title('Visão Geral (todos os polos)', fontsize=11)

# Adiciona rótulo manual para SP-1 que ficou fora do zoom
ax_full.annotate(f'SP-1: λ={lam_sp.real:.1f} s⁻¹\n(ζ=1, overdamped)',
                 xy=(lam_sp.real, 0), xytext=(lam_sp.real + 2, r_all * 0.08),
                 fontsize=9, color='tab:blue',
                 arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

# --- Painel direito: zoom na região de baixa frequência ---
_plot_poles_on_ax(ax_zoom, eigenvalues, zoom=True)
zoom_r = abs(lam_ph) * 2.8
ax_zoom.set_xlim(-zoom_r, zoom_r * 0.6)
ax_zoom.set_ylim(-zoom_r, zoom_r)
ax_zoom.set_aspect('equal')
ax_zoom.set_title('Zoom — Fugóide + SP-2', fontsize=11)

# Marcação do instável (semiplano direito sombreado)
ax_zoom.axvspan(0, zoom_r * 0.6, alpha=0.06, color='red', label='Instável (Re > 0)')
ax_zoom.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('mapa_polos.png', dpi=150)
plt.show()

# =============================================================================
# BLOCO 6 — SIMULAÇÃO TEMPORAL (Cook, Cap. 5 — solução das EOM)
#
# Entrada: degrau de 5 deg no elevator a partir de t=0.
# Simulação NÃO-LINEAR: solve_ivp integra as EOM completas de Bryan.
# Simulação LINEAR   : scipy.signal.lsim integra o sistema de espaço de estados.
#
# Condição inicial: perturbação nula (trim exato).
# As variáveis plotadas são PERTURBAÇÕES em relação ao trim.
# =============================================================================

t_end  = 60.0           # duração [s] — longa para capturar fugóide
t_eval = np.linspace(0, t_end, 3000)
delta_e_step = np.deg2rad(5.0)   # degrau de 5 deg no elevator

# Entrada temporal: degrau ligado para t>=0 (permanente)
u_input = np.zeros((len(t_eval), 2))
u_input[:, 0] = delta_e_step     # deltae
u_input[:, 1] = 0.0              # ΔT = 0

# Estado inicial: trim exato
x0_nl = [U_trim, W_trim, 0.0, theta_trim]

# --- Simulação não-linear ---
def ode_step(t, state):
    return longitudinal_nonlinear(t, state, delta_e_trim + delta_e_step, T_trim)

sol_nl = solve_ivp(ode_step, [0, t_end], x0_nl,
                   t_eval=t_eval, method='RK45',
                   rtol=1e-8, atol=1e-10)

U_nl = sol_nl.y[0]
W_nl = sol_nl.y[1]
q_nl = sol_nl.y[2]
th_nl = sol_nl.y[3]

# Perturbações em relação ao trim
u_nl   = U_nl  - U_trim
w_nl   = W_nl  - W_trim
q_nl_p = q_nl               # q_trim = 0
th_nl_p= th_nl - theta_trim
alpha_nl = np.arctan2(W_nl, U_nl) - alpha_trim  # Δalpha

# --- Simulação linear ---
t_lin, y_lin, _ = signal.lsim(sys_lin, u_input, t_eval,
                               X0=[0.0, 0.0, 0.0, 0.0])
u_lin   = y_lin[:, 0]
w_lin   = y_lin[:, 1]
q_lin_p = y_lin[:, 2]
th_lin_p= y_lin[:, 3]
alpha_lin = w_lin / V0    # Δalpha ≈ Δw/V₀ (linearização)

print("\n" + "=" * 62)
print("  BLOCO 6 — SIMULAÇÃO TEMPORAL")
print("=" * 62)
print(f"  Entrada: degrau deltae = {np.rad2deg(delta_e_step):.1f} deg (permanente)")
print(f"  Duração: {t_end:.0f} s   |   {len(t_eval)} pontos")
print(f"  Método NL : RK45 (rtol=1e-8)")
print(f"  Método Lin: scipy.signal.lsim")
print("=" * 62)

# --- Figura 2: Respostas temporais ---
fig2, axes = plt.subplots(3, 2, figsize=(15, 10))
fig2.suptitle(f'Resposta a Degrau de deltae = {np.rad2deg(delta_e_step):.0f} deg '
              f'— Dinâmica Longitudinal', fontsize=14)

pares = [
    (u_nl,     u_lin,    'Δu  [m/s]',    axes[0, 0]),
    (w_nl,     w_lin,    'Δw  [m/s]',    axes[0, 1]),
    (np.rad2deg(alpha_nl), np.rad2deg(alpha_lin), 'Δalpha  [ deg]', axes[1, 0]),
    (np.rad2deg(q_nl_p),   np.rad2deg(q_lin_p),   'q   [ deg/s]', axes[1, 1]),
    (np.rad2deg(th_nl_p),  np.rad2deg(th_lin_p),  'Δtheta  [ deg]',   axes[2, 0]),
]

for nl_data, lin_data, ylabel, ax in pares:
    ax.plot(t_eval, nl_data,  label='Não-linear', color='tab:blue')
    ax.plot(t_eval, lin_data, label='Linear',      color='tab:orange',
            linestyle='--', linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('t [s]')
    ax.legend(fontsize=9)
    ax.grid(True)

# Painel vazio (3,2) — usar para legenda ou trajetória
axes[2, 1].axis('off')
axes[2, 1].text(0.5, 0.5,
    f'Modos:\n'
    f'  SP: omegan={wn_sp:.3f} rad/s,  zeta={zeta_sp:.3f}\n'
    f'       T={T_sp:.2f} s\n\n'
    f'  PH: omegan={wn_ph:.4f} rad/s, zeta={zeta_ph:.4f}\n'
    f'       T={T_ph:.1f} s',
    transform=axes[2, 1].transAxes,
    ha='center', va='center', fontsize=11,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('simulacao_temporal.png', dpi=150)
plt.show()

# =============================================================================
# BLOCO 7 — FUNÇÕES DE TRANSFERÊNCIA E ANÁLISE DE FREQUÊNCIA (Cook, Cap. 5)
#
# Aplicando a transformada de Laplace ao sistema linearizado:
#   s.X(s) = A.X(s) + B.U(s)   →  X(s) = (sI - A)⁻1.B.U(s)
#
# As funções de transferência são:
#   H_i_j(s) = Y_i(s) / U_j(s)
#
# Aqui extraímos: theta(s)/deltae(s)  e  q(s)/deltae(s)
#
# O denominador comum é a equação característica Δ(s).
# Sua fatoração revela os dois modos longitudinais.
# =============================================================================

# FT de cada saída em relação à entrada deltae (coluna 0 de B)
num_u,   den = signal.ss2tf(A, B, C, D, input=0)   # todas as saídas, input deltae
# Índices de saída: 0=u, 1=w, 2=q, 3=theta
tf_u   = signal.TransferFunction(num_u[0], den)
tf_w   = signal.TransferFunction(num_u[1], den)
tf_q   = signal.TransferFunction(num_u[2], den)
tf_th  = signal.TransferFunction(num_u[3], den)

# Coeficientes da equação característica
print("\n" + "=" * 62)
print("  BLOCO 7 — FUNÇÕES DE TRANSFERÊNCIA (Cook, Cap. 5)")
print("=" * 62)
print(f"\n  Equação característica Δ(s) = det(sI - A):")
coefs = np.poly(A)
termos = ['s4', 's3', 's2', 's1', 's⁰']
expr = ' + '.join(f'({c:.4f}).{t}' for c, t in zip(coefs, termos))
print(f"    Δ(s) = {expr}")

# Raízes (verificação — devem coincidir com autovalores)
print(f"\n  Raízes de Δ(s): {np.roots(coefs)}")

print(f"\n  FT: theta(s)/deltae(s)")
print(f"    Numerador : {tf_th.num}")
print(f"    Denominador: {tf_th.den}")

print(f"\n  FT: q(s)/deltae(s)")
print(f"    Numerador : {tf_q.num}")
print(f"    Denominador: {tf_q.den}")
print("=" * 62)

# --- Diagramas de Bode ---
omega = np.logspace(-3, 2, 2000)   # 0.001 a 100 rad/s

w_th, mag_th, ph_th = signal.bode(tf_th, w=omega)
w_q,  mag_q,  ph_q  = signal.bode(tf_q,  w=omega)

fig3, axes3 = plt.subplots(2, 2, figsize=(15, 8))
fig3.suptitle('Diagrama de Bode — Dinâmica Longitudinal (Cook, Cap. 5)', fontsize=14)

# theta/deltae — amplitude
ax = axes3[0, 0]
ax.semilogx(w_th, mag_th, color='tab:blue')
ax.axvline(wn_sp, color='tab:red', linestyle='--', linewidth=1,
           label=f'omegan_SP = {wn_sp:.3f} rad/s')
ax.axvline(wn_ph, color='tab:green', linestyle='--', linewidth=1,
           label=f'omegan_PH = {wn_ph:.4f} rad/s')
ax.set_ylabel('|theta/deltae|  [dB]')
ax.set_title('theta(s)/deltae(s) — Amplitude')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.4)

# theta/deltae — fase
ax = axes3[1, 0]
ax.semilogx(w_th, ph_th, color='tab:blue')
ax.axvline(wn_sp, color='tab:red', linestyle='--', linewidth=1)
ax.axvline(wn_ph, color='tab:green', linestyle='--', linewidth=1)
ax.set_ylabel('∠theta/deltae  [ deg]')
ax.set_xlabel('omega [rad/s]')
ax.set_title('theta(s)/deltae(s) — Fase')
ax.grid(True, which='both', alpha=0.4)

# q/deltae — amplitude
ax = axes3[0, 1]
ax.semilogx(w_q, mag_q, color='tab:orange')
ax.axvline(wn_sp, color='tab:red', linestyle='--', linewidth=1,
           label=f'omegan_SP = {wn_sp:.3f} rad/s')
ax.axvline(wn_ph, color='tab:green', linestyle='--', linewidth=1,
           label=f'omegan_PH = {wn_ph:.4f} rad/s')
ax.set_ylabel('|q/deltae|  [dB]')
ax.set_title('q(s)/deltae(s) — Amplitude')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.4)

# q/deltae — fase
ax = axes3[1, 1]
ax.semilogx(w_q, ph_q, color='tab:orange')
ax.axvline(wn_sp, color='tab:red', linestyle='--', linewidth=1)
ax.axvline(wn_ph, color='tab:green', linestyle='--', linewidth=1)
ax.set_ylabel('∠q/deltae  [ deg]')
ax.set_xlabel('omega [rad/s]')
ax.set_title('q(s)/deltae(s) — Fase')
ax.grid(True, which='both', alpha=0.4)

plt.tight_layout()
plt.savefig('bode_longitudinal.png', dpi=150)
plt.show()

print("\n  Arquivos gerados:")
print("    mapa_polos.png")
print("    simulacao_temporal.png")
print("    bode_longitudinal.png")
print("\n  Análise concluída.")
