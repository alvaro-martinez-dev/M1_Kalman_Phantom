# Equations of motion for different regions to propagate the particle

from src.constants import KAPPA   # [MeV/(T·mm)], field.By in T, p in MeV

def eom():
    def eom_free_space(tx):
        def x_free_space(x0, tx, z, z0): #x(z) = x0 + tx(z-z0)
            return x0 + tx*(z-z0)
        return x_free_space

    def eom_magnet(field, particle, p, kappa=KAPPA): #xprime = tx = dx/dz
        def fun_magnet(z, x, xprime):
            return -kappa * (particle.charge/p) * (1+xprime**2)**(1.5) *field.By(x,z)
        return fun_magnet

    def eom_phantom():