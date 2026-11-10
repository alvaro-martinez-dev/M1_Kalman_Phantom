# Equations of motion for different regions to propagate the particle

def eom():
    def eom_free_space(tx):
        def fun_free_space(z, x):
            return tx
        return fun_free_space

    def eom_magnet(field, particle, p, kappa): #xprime = tx = dx/dz
        def fun_magnet(z, x, xprime):
            return -kappa * (particle.charge/p) * (1+xprime)**(1.5) *field.By(x,z)
        return fun_magnet

    def eom_phantom():