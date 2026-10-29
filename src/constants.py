# src/constants.py
# Units: MeV, mm, c = 1

# This constants.py has particle and material constants

from dataclasses import dataclass
import numpy as np

# ----------------------------------------------
# --------------- Constants --------------------
# ----------------------------------------------

HIGHLAND_CONST = 13.6       # MeV, multiple scattering
K_BETHE = 0.307075          # MeV cm^2 / mol (Bethe-Bloch)
M_ELECTRON = 0.51099895     # MeV

# ----------------------------------------------
# ---------------- Particles -------------------
# ----------------------------------------------

@dataclass(frozen=True)
class Particle:
    name: str
    mass: float   # MeV
    charge: int   # units of e
    ctau: float = np.inf   # mm, mean decay length in the rest frame (np.inf = stable)

# --- Leptons ---
ELECTRON  = Particle("e-",  0.51099895, -1)
POSITRON  = Particle("e+",  0.51099895, +1)
MU_MINUS  = Particle("mu-", 105.6583755, -1, 658_638.4)
MU_PLUS   = Particle("mu+", 105.6583755, +1, 658_638.4)

# --- Neutrinos (massless, invisible to the detector) ---
NU_MU     = Particle("nu_mu",    0.0, 0)
NU_MU_BAR = Particle("nu_mubar", 0.0, 0)

# --- Mesons ---
PI_MINUS  = Particle("pi-", 139.57039, -1, 7_804.5)
PI_PLUS   = Particle("pi+", 139.57039, +1, 7_804.5)
K_MINUS   = Particle("K-",  493.677, -1, 3_711.0)
K_PLUS    = Particle("K+",  493.677, +1, 3_711.0)

# --- Quarkonia ---
JPSI      = Particle("J/psi", 3096.900, 0, 2.13e-9)   # Gamma = 92.6 keV -> decays at the vertex

# --- Barions ---
PROTON     = Particle("p",    938.27208816, +1)
ANTIPROTON = Particle("pbar", 938.27208816, -1)

# --- Nucleus (iones de terapia) ---
DEUTERON = Particle("d",     1875.61294, +1)
ALPHA    = Particle("alpha", 3727.37941, +2)
CARBON12 = Particle("C12",   11174.862,  +6)

# ----------------------------------------------
# ------------ Two-body decay channels ---------
# ----------------------------------------------

# parent: decaying Particle
# daughters: the two Particles it decays into
# br: branching fraction of the channel

@dataclass(frozen=True)
class DecayChannel:
    parent: Particle
    daughters: tuple[Particle, Particle]
    br: float = 1.0

    def __post_init__(self):
        d1, d2 = self.daughters
        if self.parent.charge != d1.charge + d2.charge:
            raise ValueError(f"{self.parent.name} -> {d1.name} {d2.name} does not conserve charge")
        if self.parent.mass <= d1.mass + d2.mass:
            raise ValueError(f"{self.parent.name} -> {d1.name} {d2.name} is kinematically forbidden")

JPSI_MUMU   = DecayChannel(JPSI,     (MU_PLUS, MU_MINUS),    br=0.05961)
JPSI_EE     = DecayChannel(JPSI,     (POSITRON, ELECTRON),   br=0.05971)
PI_PLUS_MUNU  = DecayChannel(PI_PLUS,  (MU_PLUS, NU_MU),     br=0.99988)
PI_MINUS_MUNU = DecayChannel(PI_MINUS, (MU_MINUS, NU_MU_BAR), br=0.99988)
K_PLUS_MUNU   = DecayChannel(K_PLUS,   (MU_PLUS, NU_MU),     br=0.6356)
K_MINUS_MUNU  = DecayChannel(K_MINUS,  (MU_MINUS, NU_MU_BAR), br=0.6356)

#---------------------------------------------------------------------------------------------
#------------------- Phantom Materials -------------------------------------------------------
#---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    name: str
    density: float    # g/cm^3
    Z_A: float        # <Z/A>
    I: float          # mean excitation energy, MeV
    X0: float         # radiation length, mm

WATER = Material("water", 1.0, 0.55509, 75e-6, 360.8)
