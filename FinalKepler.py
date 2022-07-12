import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def norm(vec):
    return np.linalg.norm(vec)


def angle(vec):
    return np.sign(vec[1])*np.arccos(vec[0]/norm(vec))


def distance(arr, r0):
    return [norm(arr[:2, i] - r0[:2]) for i in range(len(arr[0]))]
    

def hit(t, xp):
    return not (norm(xp[1:3]) < 10**-3)

hit.terminal = True


# Return the last orbit of the planet.
def finalOrbit(ret):
    Fangle = angle(ret[1:3, -1])
    for i in range(len(ret[0, :])-2, 1, -1):
        ang0 = abs(angle(ret[1:3, i-1]) - Fangle) % (2*np.pi)
        ang1 = abs(angle(ret[1:3, i]) - Fangle) % (2*np.pi)
        ang2 = abs(angle(ret[1:3, i+1]) - Fangle) % (2*np.pi)
        if ang1< ang0 and ang1 < ang2:
            return ret[:, i:]
    return ret


# Compute <1/r>
def invR(orbit):
    N = len(orbit[0, :])
    s1 = sum((1/norm(orbit[1:3, i]) for i in range(N)))
    s2 = sum(orbit[1, i]/norm(orbit[1:3, i]) for i in range(N))
    s3 = sum(orbit[2, i]/norm(orbit[1:3, i]) for i in range(N))
    return s1/N, s2/N, s3/N


class Kepler_Orbit:
    def __init__(self, xp0, M, m, G, dp, res, method='Kepler', rDep=-2, pDep=1):
        self.xp0 = xp0
        self.res = res
        self.M = M
        self.m = m
        self.m0 = m
        self.M0 = M
        self.G = G
        self.k = -G*M*m

        self.dp = dp
        self.method = method
        self.rDep = rDep
        self.pDep = pDep

    # Compute CoM
    def CoM(self, xp, normedReturn=False):
        t = xp[0]
        x = [xp[1], xp[2], 0]
        p = [xp[3], xp[4], 0]
        if self.method == 'MassPlanet':
            self.m = self.m0*np.exp(self.dp*t)
            self.k = -self.G*self.M*self.m
        if self.method == 'MassSun':
            self.M = self.M0*np.exp(self.dp*t)
            self.k = -self.G*self.M*self.m        
        # if self.method == 'MassPlanet':
        #     self.m = self.m0
        #     self.k = -self.G*self.M*self.m
        RL = np.cross(p, np.cross(x, p)) - self.G*self.M*self.m**2*np.array(x)/norm(x)

        if normedReturn:
            return [norm(p)**2/(2*self.m) -self.G*self.M*self.m/norm(x),
                norm(np.cross(x, p)[2]),
                norm(RL),
                norm(RL)/(self.m**2*self.M*self.G)
                ]  
        else:
            return [norm(p)**2/(2*self.m) -self.G*self.M*self.m/norm(x),
                    np.cross(x, p)[2],
                    RL,
                    norm(RL)/(self.m**2*self.M*self.G)
                    ]

    def Calc(self, lam):
        res = scipy.integrate.solve_ivp(self.pdF, (0, lam), self.xp0, method='RK45', events = hit, dense_output=True, 
                                        vectorized=False, max_step=self.res)
        return res.y

    def pdF(self, lam, xp):
        t = xp[0]
        x = xp[1:3]
        p = xp[3:5]
        xx=norm(x)
        pp = norm(p)
        if self.method == 'Radial':
            return np.array((1, 
                            p[0]/self.m, 
                            p[1]/self.m,
                            self.k *x[0]/xx**3 + self.dp*self.m*x[0]*xx**(-1+self.rDep),
                            self.k *x[1]/xx**3 + self.dp*self.m*x[1]*xx**(-1+self.rDep),
                            ))

        if self.method == 'Drag': 
            return np.array((1, 
                            p[0]/self.m, 
                            p[1]/self.m,
                            self.k * x[0] / xx**3 + self.dp*p[0]/self.m,
                            self.k * x[1] / xx**3 + self.dp*p[1]/self.m,
                            ))   
        if self.method == 'MassPlanet':
            self.m = self.m0*np.exp(self.dp*t)
            self.k = -self.G*self.M*self.m
            return np.array((1, 
                            p[0]/self.m, 
                            p[1]/self.m,
                            self.k *x[0]/xx**3,
                            self.k *x[1]/xx**3,
                            )) 

        if self.method == 'MassSun':
            self.M = self.M0*np.exp(self.dp*t)
            self.k = -self.G*self.M*self.m
            return np.array((1, 
                            p[0]/self.m, 
                            p[1]/self.m,
                            self.k *x[0]/xx**3,
                            self.k *x[1]/xx**3,
                            ))                

    def setupOrbit(self, T):
        Ret = self.Calc(T)
        lRet = finalOrbit(Ret)

        r0 = [min(lRet[1])/2+ max(lRet[1])/2, min(lRet[2])/2 + max(lRet[2])/2]
        rmin = min(distance(lRet[1:3], [0, 0]))
        rmax = max(distance(lRet[1:3], [0, 0]))

        print('final pos', Ret[:, -1])
        print('E, L, A', self.CoM(Ret[:, -1]))
        print('r+- and e', rmin, rmax, (rmax-rmin)/(rmax+rmin))
        print('x+- and z+-', min(lRet[1]), max(lRet[1]), min(lRet[2]), max(lRet[2]), r0)
        return Ret


def ComparePlot(T, xp0, M, m, res, dpArr, method = 'Radial'):
    Orbits = []
    fig, ax = plt.subplots(max(len(dpArr), 2), 1)
    for i, dp in enumerate(dpArr):
        orbit = Kepler_Orbit(xp0, M, m, G, dp, res) #self, xp0, M, m, G, dp, res
        ret = orbit.setupOrbit(T)
        Orbits.append(orbit)

        ax[i].plot(ret[1], ret[2], label = f'dp: {dpArr[i]}')

        ax[i].legend()     
        ax[i].set_title("Orbits with different coefficients for a $r^{0}$ Force")
    plt.show()


def CoMplot(T, xp0, M, m, res, dpArr, method='Radial', rDep=-2, pDep=1):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)

    fig, ax = plt.subplots(2, figsize=(10, 9), tight_layout=True)

    orbit = Kepler_Orbit(xp0, M, m, G, dpArr[0], res, method, rDep=rDep, pDep=pDep) #self, xp0, M, m, G, dp, res
    ret = orbit.setupOrbit(T)
    print('ret0', ret[:, 0])
    CoM0 = orbit.CoM(ret[:, 0], normedReturn=True)[:3]
    print('Com0', CoM0)
    energyArr = np.array([orbit.CoM(ret[:, i])[0] for i in range(len(ret[0]))])
    angMomArr = np.array([norm(orbit.CoM(ret[:, i])[1]) for i in range(len(ret[0]))])
    RLArr = np.array([norm(orbit.CoM(ret[:, i])[2]) for i in range(len(ret[0]))])

    lOrbit = finalOrbit(ret)
    print(lOrbit[:, 0])
    print('1/r', invR(lOrbit))

    
    if orbit.method == "Radial":
        ax[1].plot(ret[0], np.add(energyArr, -orbit.CoM(ret[:, 0])[0]), label='$E - E_0$')
        ax[1].plot(ret[0], np.add(angMomArr, -norm(orbit.CoM(ret[:, 0])[1])), label='$L - L_0$')
        ax[1].plot(ret[0], np.add(RLArr, -norm(orbit.CoM(ret[:, 0])[2])), label='$A - A_0$')
        ForceStr = f'$F_{{{orbit.method}}} = {orbit.dp}r^{{{orbit.rDep}}}$'


    elif orbit.method == 'Drag':
        ForceStr = f'$F_{{{orbit.method}}} = {orbit.dp}p^{{{orbit.pDep}}}$'        
        ax[1].plot(ret[0], (1/CoM0[0])*energyArr, label='$E/E_0$')
        ax[1].plot(ret[0], (1/CoM0[1])*angMomArr, label='$L/L_0$')
        ax[1].plot(ret[0], (1/CoM0[2])*RLArr, label='$A/A_0$')

        energyFit = np.exp(-2*dpArr[0] * ret[0])
        angMomFit = np.exp(dpArr[0] * ret[0])
        ax[1].grid(True)
        ax[1].plot(ret[0], energyFit, label='E_fit', linestyle='dashed', color='purple')
        ax[1].plot(ret[0], angMomFit, label='L_fit', linestyle='dashed', color='lime')
        ax[1].set_yscale('log')

    elif orbit.method == 'MassPlanet':
        ax[1].plot(ret[0], energyArr, label='$E$')
        ax[1].plot(ret[0], angMomArr, label='$L$')
        ax[1].plot(ret[0], RLArr, label='$A$')

        energyFit = CoM0[0]*np.exp(3*dpArr[0] * ret[0])
        RLFit = CoM0[2]*np.exp(2*orbit.dp*ret[0])
        ax[1].plot(ret[0], energyFit, label='E_fit', linestyle='dashed', color='purple')
        ax[1].plot(ret[0], RLFit, label='A_fit', linestyle='dashed', color='lime')
        ForceStr = f'$\dot m = {orbit.dp}m$'

    elif orbit.method == 'MassSun':
        ax[1].plot(ret[0], energyArr, label='$E$')
        ax[1].plot(ret[0], angMomArr, label='$L$')
        ax[1].plot(ret[0], RLArr, label='$A$')

        energyFit = CoM0[0]*np.exp(2*dpArr[0] * ret[0])
        RLFit = CoM0[2]*np.exp(1*orbit.dp*ret[0])
        ax[1].plot(ret[0], energyFit, label='E_fit', linestyle='dashed', color='purple')
        ax[1].plot(ret[0], RLFit, label='A_fit', linestyle='dashed', color='lime')
        ForceStr = f'$\dot M = {orbit.dp}M$'

    ax[0].set_title(f"Kepler orbit with perturbation {ForceStr}")
    ax[0].plot(ret[1], ret[2], label = f'$x_0={xp0[1:3]},\ p_0={xp0[3:5]}$ and $E_0, L_0, A_0 = {CoM0}$')
    ax[0].plot(lOrbit[1], lOrbit[2], color='purple', label='Last orbit')
    ax[0].legend(loc=1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    ax[1].set_title(f'Variation in the constants of motion due to perturbative force {ForceStr}')
    ax[1].set_xlabel('time $t$')
    ax[1].legend(loc=1)
    plt.savefig(f'Kepler{orbit.method}_dp{orbit.dp}rpDep{orbit.rDep}{orbit.pDep}.pdf', bbox_inches='tight')
    plt.show()


MBH = [1]
NBH = len(MBH)
PosBH = np.array([(0, 0)])

G = 1.0
M = 1.0
m = 1.0
x0 = (0, 0.5)
xp0 = (0, x0[0], x0[1], 1, 0.0)


# CoMplot(1, xp0, M, m, 0.01, [-0.001], method='Kepler', rDep=-2)


# Plots for MassPlanet perturbations
CoMplot(50, xp0, M, m, 0.01, [1e-2], method='MassSun')


# Plots for MassPlanet perturbations
# CoMplot(100, xp0, M, m, 0.01, [1e-3], method='MassPlanet')


# Plots for Drag perturbations:
# CoMplot(100, xp0, M, m, 0.01, [-0.01], method='Drag', pDep=0)
# CoMplot(100, xp0, M, m, 0.01, [-0.001], method='Drag', pDep=0)

# CoMplot(100, xp0, M, m, 0.01, [-0.01], method='Drag')
# CoMplot(100, xp0, M, m, 0.01, [-0.001], method='Drag')

# CoMplot(100, xp0, M, m, 0.01, [-0.01], method='Drag', pDep=2)
# CoMplot(100, xp0, M, m, 0.01, [-0.001], method='Drag', pDep=2)


# Plots for Radial perturbations:
# CoMplot(100, xp0, M, m, 0.01, [-0.01], method='Radial', rDep=-3)
# CoMplot(100, xp0, M, m, 0.01, [-0.1], method='Radial', rDep=-2)
# CoMplot(100, xp0, M, m, 0.01, [-0.1], method='Radial', rDep=-1)
# CoMplot(100, xp0, M, m, 0.01, [-0.1], method='Radial', rDep=-1.5)
# CoMplot(100, xp0, M, m, 0.01, [-0.1], method='Radial', rDep=-0.5)
# CoMplot(100, xp0, M, m, 0.01, [-0.1], method='Radial', rDep=-0)


# TODO 360 flux, mass density plot, keep track of momentum of both BHs  
# TODO: comparison function for perturbed and unperturbed
