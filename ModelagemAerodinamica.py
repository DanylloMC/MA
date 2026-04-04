import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'figure.figsize': (12, 6)
})

# =============================================================================
# BLOCO 1 — PARÂMETROS DA AERONAVE (F-4C Phantom)
# Todos os valores em unidades SI
# =============================================================================

# --- Parâmetros físicos ---
m   = 14.0      # Massa total [kg]
Iy  = 0.385    # Momento de inércia em arfagem [kg·m²]  SEGUNDO XFLR5 = 0.835
g   = 9.81     # Aceleração gravitacional [m/s²]
S   = 1.213    # Área da asa [m²]
c   = 0.5048   # Corda aerodinâmica média (c̄) [m]

# --- Condição de voo de referência (trim) ---
V0      = 12.0    # Velocidade de trim [m/s]
rho     = np.rho   # Densidade do ar ao nível do mar [kg/m³]
theta_e = np.deg2rad(9.4)   # Ângulo de arfagem de trim [rad] (= αe pois γe=0)
alpha_e = theta_e            # Ângulo de ataque de trim [rad]

# Velocidades de trim em eixos de corpo
Ue = V0 * np.cos(theta_e)   # [m/s]
We = V0 * np.sin(theta_e)   # [m/s]

# Pressão dinâmica de trim
q_dyn_e = 0.5 * rho * V0**2  # [Pa]

# --- Derivadas adimensionais de estabilidade longitudinal ---
# Fonte: Exemplo 4.2, Cap. 4
CX_u     =  0.0076
CX_w     =  0.0483
CX_wdot  =  0.0
CX_q     =  0.0
CX_de    =  0.0618    # δe = elevator

CZ_u     = -0.7273
CZ_w     = -3.1245
CZ_wdot  = -0.3997
CZ_q     = -1.2109
CZ_de    = -0.3741

Cm_u     =  0.0340
Cm_w     = -0.2169
Cm_wdot  = -0.5910
Cm_q     = -1.2732
Cm_de    = -0.5581

# Polar de arrasto (estimativas típicas F-4C)
CD0  = 0.0806    # Coeficiente de arrasto parasita (Analítico)
k    = 0.0706    # Fator de arrasto induzido (1/(pi*AR*e)) (K da ASA)
CL_e = (m * g) / (q_dyn_e * S)  # CL de trim (equilíbrio)

# Deflexão de elevator e empuxo no trim (valores a serem determinados no trim)
delta_e_e = 0.0    # será calculado no Bloco 2
T_e       = 0.0    # será calculado no Bloco 2

print("=" * 60)
print("  PARÂMETROS DA AERONAVE — F-4C Phantom")
print("=" * 60)
print(f"  Massa                : {m:.0f} kg")
print(f"  Inércia em arfagem   : {Iy:.0f} kg·m²")
print(f"  Área da asa          : {S:.3f} m²")
print(f"  Corda média          : {c:.3f} m")
print(f"  Velocidade de trim   : {V0:.1f} m/s")
print(f"  Densidade do ar      : {rho:.4f} kg/m³")
print(f"  Pressão dinâmica     : {q_dyn_e:.1f} Pa")
print(f"  Ângulo de trim θe    : {np.rad2deg(theta_e):.1f}°")
print(f"  Ue                   : {Ue:.2f} m/s")
print(f"  We                   : {We:.2f} m/s")
print(f"  CL de trim           : {CL_e:.4f}")
print("=" * 60)

# =============================================================================
# BLOCO 2 — MODELO AERODINÂMICO E EQUAÇÕES DE MOVIMENTO NÃO-LINEARES
# =============================================================================

def aero_forces_moments(U, W, q, delta_e, rho=rho):
    """
    Calcula forças e momento aerodinâmicos em eixos de corpo.

    Modelo não-linear com coeficientes dependentes de:
      - Ângulo de ataque α = arctan(W/U)
      - Taxa de arfagem adimensional q̂ = q*c/(2*V)
      - Deflexão do elevator δe

    Retorna: X_aero [N], Z_aero [N], M_aero [N·m]
    """
    V     = np.sqrt(U**2 + W**2)
    alpha = np.arctan2(W, U)            # Ângulo de ataque [rad]
    q_hat = q * c / (2.0 * V + 1e-6)   # Adimensionalização (evita div/0)
    q_dyn = 0.5 * rho * V**2            # Pressão dinâmica [Pa]

    # Desvios em relação ao trim (para uso nas derivadas lineares)
    delta_u = U - Ue
    delta_w = W - We

    # --- Coeficientes em eixos de corpo (formulação perturbação + base) ---
    # CX (força axial, sentido positivo para frente)
    CX = (CX_u * delta_u / V0 +
          CX_w * delta_w / V0 +
          CX_q * q_hat +
          CX_de * delta_e)

    # CZ (força normal, sentido positivo para baixo em eixo de corpo)
    CZ = (CZ_u * delta_u / V0 +
          CZ_w * delta_w / V0 +
          CZ_q * q_hat +
          CZ_de * delta_e)

    # Cm (momento de arfagem, positivo nose-up)
    Cm = (Cm_u * delta_u / V0 +
          Cm_w * delta_w / V0 +
          Cm_q * q_hat +
          Cm_de * delta_e)

    X_aero = q_dyn * S * CX
    Z_aero = q_dyn * S * CZ
    M_aero = q_dyn * S * c * Cm

    return X_aero, Z_aero, M_aero


def longitudinal_nonlinear(t, state, delta_e, T, rho=rho):
    """
    Equações de movimento longitudinais NÃO-LINEARES (3 DOF + cinemática).

    Estado: x = [U, W, q, θ]

    Equações (Bryan, cap. 4, eq. 4.12 e 4.19 com V=p=r=0):
      m*(U̇ + q*W) = X_aero + T - m*g*sin(θ)
      m*(Ẇ - q*U) = Z_aero     + m*g*cos(θ)
      Iy*q̇        = M_aero
      θ̇           = q

    Parâmetros:
      delta_e : deflexão do elevator [rad]
      T       : empuxo [N]
    """
    U, W, q, theta = state

    X_aero, Z_aero, M_aero = aero_forces_moments(U, W, q, delta_e, rho)

    # Equação da força axial
    U_dot = (-q * W
             - g * np.sin(theta)
             + (X_aero + T) / m)

    # Equação da força normal
    W_dot = (q * U
             + g * np.cos(theta)
             + Z_aero / m)

    # Equação de momento de arfagem
    q_dot = M_aero / Iy

    # Equação cinemática de atitude
    theta_dot = q

    return [U_dot, W_dot, q_dot, theta_dot]

# =============================================================================
# BLOCO 3 — BUSCA DO PONTO DE EQUILÍBRIO (TRIM)
# =============================================================================

def trim_equations(x_opt, V_trim, gamma_trim=0.0, rho=rho):
    """
    Sistema de equações para o ponto de equilíbrio (trim).

    Variáveis de otimização: x_opt = [alpha, delta_e, T]
    Condição de trim (ṡtate = 0 para U̇=0, Ẇ=0, q̇=0):
      - q = 0, θ = alpha + gamma_trim
      - As três equações de estado devem ser zero
    """
    alpha, delta_e, T = x_opt

    theta = alpha + gamma_trim
    U = V_trim * np.cos(alpha)
    W = V_trim * np.sin(alpha)
    q = 0.0

    derivs = longitudinal_nonlinear(0, [U, W, q, theta], delta_e, T, rho)

    return [derivs[0], derivs[1], derivs[2]]


print("\n" + "=" * 60)
print("  BLOCO 3 — PONTO DE EQUILÍBRIO (TRIM)")
print("=" * 60)

# Chute inicial para [alpha, delta_e, T]
x0_trim = [alpha_e, np.deg2rad(-2.0), m * g * 0.3]

trim_sol = fsolve(
    trim_equations,
    x0_trim,
    args=(V0, 0.0, rho),
    full_output=True
)

alpha_trim  = trim_sol[0][0]
delta_e_trim = trim_sol[0][1]
T_trim      = trim_sol[0][2]
info        = trim_sol[1]

# Atualiza parâmetros globais de trim
delta_e_e = delta_e_trim
T_e       = T_trim

U_trim = V0 * np.cos(alpha_trim)
W_trim = V0 * np.sin(alpha_trim)
theta_trim = alpha_trim  # γ = 0

residuo = longitudinal_nonlinear(
    0, [U_trim, W_trim, 0.0, theta_trim], delta_e_trim, T_trim
)

print(f"\n  Solução de Trim:")
print(f"    α  (ângulo de ataque)  : {np.rad2deg(alpha_trim):+.4f}°")
print(f"    δe (elevator)          : {np.rad2deg(delta_e_trim):+.4f}°")
print(f"    T  (empuxo)            : {T_trim:+.2f} N")
print(f"    U_trim                 : {U_trim:.2f} m/s")
print(f"    W_trim                 : {W_trim:.2f} m/s")
print(f"    θ_trim                 : {np.rad2deg(theta_trim):.4f}°")
print(f"\n  Resíduos (devem ser ≈ 0):")
print(f"    U̇  : {residuo[0]:.2e} m/s²")
print(f"    Ẇ  : {residuo[1]:.2e} m/s²")
print(f"    q̇  : {residuo[2]:.2e} rad/s²")
print("=" * 60)