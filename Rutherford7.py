from re import A, X
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate 
from datetime import datetime

def norm(vec):
    return np.linalg.norm(vec)

def RotMat(th):
    return np.array([[np.cos(th), -np.sin(th)],
              [np.sin(th),  np.cos(th)]])

def convert2(arr): # convert Cartesian to Spherical
    x0 = arr[1:3]
    return np.array([
        arr[0],
        norm(x0),
        np.sign(x0[1])*np.arccos(x0[0]/norm(x0))    
    ])

class BH_Orbit:
    def __init__(self, MBH, PosBH, v=0, nDilation=False, method=1, dlam=0.1):
        self.dlam = dlam # Max step size of EoM numerical integration

        self.method= method #1: Ruther, 2:Extremal, 3:Combined, 4:Schwarzschild(Unused for project, might not be fully correct)
        self.MBH = MBH
        self.NBH = len(MBH)
        self.PosBH = PosBH
        self.sep = norm(PosBH[0] - PosBH[1])
        self.v = v # Velocity of the BHs, meant to represent Kepler orbits on short timescales
        self.vel = 1 # Velocity of the incoming particle, c=1

        if method == 3 or method == 4:
            self.vel = 0.1
            self.dlam = 0.05
            self.E = 1/np.sqrt(1-self.vel**2)*np.sqrt(1-2/(0.5*self.sep)) # Energy modified to incorporate relativistic corrections.
            # self.L = 0 
            self.Rdir = -1 # Assume for Schwarzschild method that the radius is descreasing.
            self.state = 0 # Keeps track of whether the particle is close to a BH, zero means no, i>0 means close to the ith BH.
            self.switch = 100 # Within this distance, the particle is close to the BH.
        if nDilation:
            self.nDilation = nDilation
        else:
            self.nDilation = 2

    def event_Newton2Schwarz(self, t, xp): # Determine when to switch from Newtonian gravity to the Schwarzschild metric.
        x = xp[1:3]
        p = xp[3:5]
        for i in range(self.NBH):        
            if norm(xp[1:3]- self.PosBH[i]) < self.switch:
                return False
            if norm(x) > 1.2*1e4 and x[0]*p[0] + x[1]*p[1] > 0:
                return False
            if norm(x) > 1e5:
                return False
        for i in range(self.NBH):
            if not(norm(x-self.PosBH[i]) > 1.2e4 and (x[0]-self.PosBH[i, 0])*p[0] + (x[1]-self.PosBH[i, 1])*p[1] > 0):
                return True
        return False


    def event_Schwarz2Newton(self, t, xp): # Determine when to switch from the Schwarzschild metric to Newtonian gravity.     
        if xp[1] > 1.1*self.switch:
            return False
        if xp[1] < (2+1e-4)*self.MBH[self.state-1]:
            return False
        return True

    event_Newton2Schwarz.terminal = True
    event_Schwarz2Newton.terminal = True


    def ProxCheck(self, xp0):
        Prox = [(norm(xp0[1:3]-self.PosBH[0]) < self.switch), (norm(xp0[1:3]-self.PosBH[1]) < self.switch)]
        if Prox[0]:
            return 1
        elif Prox[1]:
            return 2
        else:
            return 0

    def CalcCombined(self, lam, xp0, dLam): # Does the numerical integration for method 3.
        TotalResFlag = False
        self.state = self.ProxCheck(xp0) 

        for n in range(10):
            # print('n', n, xp0, self.state)

            if self.state == 0: # Newtonian
                res = scipy.integrate.solve_ivp(self.dF1, (0, lam), xp0, method='LSODA', events = self.event_Newton2Schwarz, dense_output=True, vectorized=False,  max_step=self.dlam)
                xp0 = res.y[:, -1]

                if not TotalResFlag:
                    TotalResFlag = True
                    TotalRes = res.y 
                else:
                    TotalRes = np.append(TotalRes, res.y, axis=1)
                
                self.state = self.ProxCheck(xp0)
                if norm(xp0[1:3]) > 1e4: # Particle has escaped
                    break
            else: # Schwarzschild
                pi = np.array(xp0[3:5])
                xs = xp0[1:3] - self.PosBH[self.state-1]
                xpSpher = [xp0[0], norm(xs), np.sign(xs[1])*np.arccos(xs[0]/norm(xs))]

                self.L = (- xs[1]*pi[0] + xs[0]*pi[1])*1/np.sqrt(1-self.vel**2) # Gamma factor for special relativistic correction
                self.Rdir = -1 # Decreasing radial radius

                # print('Input for Schwarz', xs, xp0, pi, self.L)
                # print('xpSPher', xpSpher)
                res = scipy.integrate.solve_ivp(self.dF4, (0, lam), xpSpher, method='LSODA', events = self.event_Schwarz2Newton, dense_output=True, vectorized=False,  max_step=self.dlam)
                cartesianRes = np.append(self.convert(res.y), np.zeros((6, len(res.y[0, :]))), axis=0) 

                if not TotalResFlag:
                    TotalResFlag = True
                    TotalRes = cartesianRes 
                else:
                    TotalRes = np.append(TotalRes, cartesianRes, axis=1)

                xpf = cartesianRes[:, -1]
                if self.GRIdentify(xpf) != 0: # Check whether the particle has been absorbed by a black hole, if so give the particle's momentum to the BH.
                    # print('absorbed', xp0)
                    # TotalRes[5:9, -1] = xp0[5:9]
                    if self.state == 1:
                        TotalRes[3:9, -1] += [0, 0, xp0[5] + xp0[3], xp0[6]+xp0[4], xp0[7], xp0[8]]
                    else:
                        TotalRes[3:9, -1] += [0, 0, xp0[5], xp0[6], xp0[7] + xp0[3], xp0[8]+xp0[4]]
                    break
                
                # The particle has left the circle surrounding the BH and now Newtonian gravity has kicked in.
                ProxCheck = [(norm(xpf[1:3]-self.PosBH[0]) < self.switch), (norm(xpf[1:3]-self.PosBH[1]) < self.switch)]    
                if not (ProxCheck[0] or ProxCheck[1]):

                    dxp = self.dF4(lam, res.y[:, -1], timeDilation=False)
                    pf = self.convert2Cartesian(xpf, dxp[:3])[1:3]
                    # print('pf', pf, dxp)
                    xpf[3:5] = pf
                    xpf[5:] = xp0[5:]
                    if self.state==1:
                        xpf[5:7] += pi-pf
                    else:
                        xpf[7:9] += pi-pf
                    # TotalRes[:, -1] = xpf
                    self.state = 0
                    xp0 = xpf
                    # print("I don't know")
        return TotalRes


    def dF4(self, lam, xp, timeDilation = True): # Schwarzschild
        r = xp[1]
        dr2 = abs(self.E**2 - (1 + self.L**2/r**2) * (1 - 2*self.MBH[self.state-1]/r))

        if self.Rdir == -1 and abs(dr2) < 10**-5:
            # print('peri', xp)
            self.Rdir = 1
        if timeDilation:
            dt = max(r/(10*self.vel), 1)
        else:
            dt = 1
        return np.array([
            dt*self.E/(1-2*self.MBH[self.state-1]/r),
            dt*self.Rdir* np.sqrt(dr2),
            dt*self.L/r**2
        ])


    def convert2Cartesian(self, xp, dSpher):
        dr = dSpher[1]
        r = norm(xp[1:3]-self.PosBH[self.state-1])
        ret = np.zeros(9)
        ret[0] = dSpher[0]
        ret[1] = (xp[1]-self.PosBH[self.state-1, 0])/r*dr -(xp[2] - self.PosBH[self.state-1, 1])/r**2 * self.L
        ret[2] = (xp[2]-self.PosBH[self.state-1, 1])/r*dr + (xp[1] - self.PosBH[self.state-1, 0])/r**2 * self.L
        return ret

    def dF2(self, lam, xp): # Extremal black hole
        t = xp[0]
        x = xp[1:3]
        p = xp[3:5]

        r = np.zeros(self.NBH)
        q = np.zeros((self.NBH, 2))
        for i in range(self.NBH):
            r[i] = np.sqrt((x[0] - self.PosBH[i, 0])**2 + (x[1] - (self.PosBH[i, 1] + (-1)**i*self.v*(t-1e4)))**2)

            q[i] = np.array([(x[0] -  self.PosBH[i, 0]), x[1] - (self.PosBH[i, 1] + (-1)**i*self.v*(t-1e4))])
        a =0.001
        dt = max(1/(a**(1/self.nDilation)+1/r[0] + 1/r[1])**self.nDilation, 0.1)

        return np.array([dt*self.U(x)**2, 
                        dt*self.U(x)**-2*p[0], 
                        dt*self.U(x)**-2*p[1], 
                        dt*2*self.U(x)*self.dUdxi(x, 0),
                        dt*2*self.U(x)*self.dUdxi(x, 1),
                        dt*-2*self.U(x)*self.dUdxi2(x, 0, 0),
                        dt*-2*self.U(x)*self.dUdxi2(x, 1, 0),
                        dt*-2*self.U(x)*self.dUdxi2(x, 0, 1),
                        dt*-2*self.U(x)*self.dUdxi2(x, 1, 1),
                        ])
        
    def U(self, x):
        return 1 + sum(self.MBH[i]/norm(x-self.PosBH[i]) for i in range(self.NBH))


    def dUdxi(self, x, j):
        return (sum(self.MBH[i]*(self.PosBH[i, j] - x[j])/norm(x-self.PosBH[i])**3 for i in range(self.NBH)))

    def dUdxi2(self, x, j, i):
        return self.MBH[i]*(self.PosBH[i, j] - x[j])/norm(x-self.PosBH[i])**3


    def dF1(self, lam, xp): #Rutherford Scattering
        t = xp[0]
        x = xp[1:3]
        p = xp[3:5]
        r = np.zeros(self.NBH)
        q = np.zeros((self.NBH, 2))
        for i in range(self.NBH):
            r[i] = np.sqrt((x[0] - self.PosBH[i, 0])**2 + (x[1] - (self.PosBH[i, 1] + (-1)**i*self.v*(t-1e4)))**2)

            q[i] = np.array([(x[0] -  self.PosBH[i, 0]), x[1] - (self.PosBH[i, 1] + (-1)**i*self.v*(t-1e4))])
        a = self.vel*0.001
        dt = 1/(a**(1/self.nDilation)+1/r[0] + 1/r[1])**self.nDilation
    

        return np.array([dt,
                        dt*p[0],
                        dt*p[1],
                        dt*sum([-self.MBH[i]*q[i, 0] / r[i]**3 for i in range(self.NBH)]),
                        dt*sum([-self.MBH[i]*q[i, 1] / r[i]**3 for i in range(self.NBH)]),
                        dt*self.MBH[0] * q[0, 0]/r[0]**3,
                        dt*self.MBH[0] * q[0, 1]/r[0]**3,
                        dt*self.MBH[1] * q[1, 0]/r[1]**3,
                        dt*self.MBH[1] * q[1, 1]/r[1]**3])



    def GRmomentum(self, xpi, xpf): # Calculate Final momentum for various methods
        xpf1 = np.array(xpf) 
        PPPi = (xpi[3]**2+xpi[4]**2)**0.5
        PPPf = ((xpf[3]+xpf[5] + xpf[7])**2+(xpf[4]+xpf[6]+xpf[8])**2)**0.5
        xpi = (1/PPPi)*np.array(xpi)
        xpf = (1/PPPf)*np.array(xpf)

        xi = xpi[1:3]
        pf = xpf[3:5]
        if self.method == 1 or self.method == 3:   
            return [xpf1[3:5], xpf1[5:7],  xpf1[7:], xpf1[3:5] + xpf1[5:7] + xpf1[7:]]
        if self.method == 2:
            return [pf/self.U(xi)**2, xpf[5:7]/self.U(xi)**2,  xpf[7:]/self.U(xi)**2, pf/self.U(xi)**2 + xpf[5:7]/self.U(xi)**2 + xpf[7:]/self.U(xi)**2]

    def GRenergy(self, xp):
        if self.method == 1 or self.method == 3:
            return 0.5*(norm(xp[3:5])**2 - 1)
        if self.method == 2:
            U0 = self.U(xp[1:3])
            return 0.5*(norm(xp[3:5])**2/U0**2 - U0**2)


    def GRangle(self, xpf): # Calculate the asymptotic direction
        t = xpf[0]
        xf = xpf[1:3]
        pf = xpf[3:5]
        if self.method == 1 or self.method == 3 or self.method == 4:
            return np.sign(pf[1])*np.arccos(pf[0]/(pf[0]**2 + pf[1]**2)**0.5)
        if self.method == 2:
            return np.sign(pf[1])*np.arccos(pf[0]/(t*self.U(xf)**2)) # Might be wrong


    def GRCalc(self, lam, xp0, dLam): # Perform numerical integration to find the EoMs under various types of scattering.
        if self.method == 1:
            res = scipy.integrate.solve_ivp(self.dF1, (0, lam), xp0, method='LSODA', events = self.GRhit, dense_output=True, vectorized=False,  max_step=dLam)
        if self.method == 2:
            res = scipy.integrate.solve_ivp(self.dF2, (0, lam), xp0, method='LSODA', events = self.GRhit, dense_output=True, vectorized=False,  max_step=dLam)
        if self.method == 3:
            return self.CalcCombined(lam, xp0, dLam)
        if self.method == 4: # Unfinished
            print('xp0', xp0)
            xpSpher = [xp0[0], norm(xp0[1:3]), np.sign(xp0[2])*np.arccos(xp0[1]/norm(xp0[1:3]))]
            print('xpSpher', xpSpher)
            self.state = 1
            pi = np.array(xp0[3:5])
            xs = xp0[1:3] - self.PosBH[self.state-1]
            self.L = (- xs[1]*pi[0] + xs[0]*pi[1])*1/np.sqrt(1-self.vel**2)
            res = scipy.integrate.solve_ivp(self.dF4, (0, lam), xpSpher, method='RK45', events = self.GRhit, dense_output=True, vectorized=False,  max_step=dLam)
            arr =  self.convert(res.y)
            print('arr', arr)
            return np.append(arr, np.zeros((6, len(arr[0, :]))), axis=0)
        return res.y

    def GRhit(self, t, xp): # Stops the simulations in case that the particle hits a BH.
        x = xp[1:3]
        p = xp[3:5]
        for i in range(self.NBH):        
            if self.method == 1 and norm(xp[1:3]- self.PosBH[i]) < 10e-4:
                return False
            if self.method ==2 and norm(xp[1:3]- self.PosBH[i]) < 10e-2:
                return False
            if (self.method == 3 or self.method == 4) and xp[1] < self.MBH[i]*(2+0.1):
                return False
        for i in range(self.NBH):
            if not(norm(x-self.PosBH[i]) > 1.2e4 and (x[0]-self.PosBH[i, 0])*p[0] + (x[1]-self.PosBH[i, 1])*p[1] > 0):
                return True
        return False

    GRhit.terminal = True

    def GRIdentify(self, xpf): # Identify whether the particle is close to a black hole.
        R = 0.5
        if self.method != 3:        
            for i in range(self.NBH):    
                if norm(xpf[1:3]- self.PosBH[i]) <R:
                    return i + 1 # Return ith BH
        else:
            for i in range(self.NBH):    
                if norm(xpf[1:3]- self.PosBH[i]) <(2+R)*self.MBH[i]:
                    return i + 1 # Return ith BH
        return 0

    def initCond(self, B, R, ang): # Convert initial conditions in terms of scattering theory into cartesian coordinates, with relativistic correction terms.
        x0 = (-R*np.cos(ang)-B*np.sin(ang), -R*np.sin(ang) + B*np.cos(ang))
        if self.method == 1 or self.method == 3:
            v0 = self.vel*np.array((np.cos(ang), np.sin(ang)))
            xp0 = (0, x0[0], x0[1], v0[0], v0[1]) + (0, 0, 0, 0)
            return xp0
        elif self.method == 2:
            U0 = self.U(x0)
            v0 = (U0*np.cos(ang), U0*np.sin(ang))
            xp0 = (0, x0[0], x0[1], v0[0], v0[1]) + (0, 0, 0, 0)

        elif self.method == 4:
            v0 = self.vel/(1-2*self.MBH[self.state-1]/norm(x0))*np.array([np.cos(ang), np.sin(ang)])
            xp0 = (0, x0[0], x0[1], v0[0], v0[1]) + (0, 0, 0, 0)
        return xp0

    
    def SimMomentum(self, B, T, dlam, ang=0): # Compute the momentum transfer of a particle to both black holes.
        R = 1e4
        xp0 = self.initCond(B, R, ang)
        Ret = self.GRCalc(T, xp0, dlam)
        dP = self.GRmomentum(xp0, Ret[:, -1])        
        cap = self.GRIdentify(Ret[:, -1]) # Was it captured by a BH, if so which?

        if cap != 0:
            if self.method == 1: # No collisions should happen in Rutherford scattering, so this is caused by inaccurate simulations
                return np.zeros(4)
            if self.method == 2 or self.method == 3 or self.method == 4:
                if cap == 1:
                    return np.array([dP[1][0]+dP[0][0], dP[2][0], dP[1][1]+dP[0][1], dP[2][1]])
                if cap == 2:
                    return np.array([dP[1][0], dP[2][0]+dP[0][0], dP[1][1], dP[2][1]+dP[0][1]])
        else:
            P = np.array([dP[1][0], dP[2][0], dP[1][1], dP[2][1]])
            # return dP[1][0], dP[2][0]
            return P

    def convert(self, arr): # Spherical to Cartesian
        if len(np.shape(arr)) == 1:
            return np.array([arr[0], 
                            arr[1] * np.cos(arr[2])+ self.PosBH[self.state-1, 0], 
                            arr[1] * np.sin(arr[2])+ self.PosBH[self.state-1, 1]])
        arr = np.array(arr)
        ret = np.zeros(np.shape(arr))
        for i in range(len(arr[0])):

            ret[0, i] = arr[0, i]
            ret[1, i] = arr[1, i] * np.cos(arr[2, i]) + self.PosBH[self.state-1, 0]
            ret[2, i] = arr[1, i] * np.sin(arr[2, i]) + self.PosBH[self.state-1, 1]
            
        return ret


def setupPlot(T, B, R, ang, dlam, MBH, PosBH, plot = True, nDilation=False, method=1): # Set up plot of a single orbit.
    BHs = BH_Orbit(MBH, PosBH, nDilation=nDilation, method=method)  
    xp0 = BHs.initCond(B, R, ang)
    t0 = datetime.now()
    Ret = BHs.GRCalc(T, xp0, dlam)
    print(datetime.now()-t0)
    print('final pos', Ret[:, -1])
    print('final mom', BHs.GRmomentum(xp0, Ret[:, -1]))
    print('Identify', BHs.GRIdentify(Ret[:, -1]))
    print('Energy', BHs.GRenergy(Ret[:, -1]))
    print('Angle', BHs.GRangle(Ret[:, -1]))
    if plot:
        plt.scatter(PosBH[:, 0], PosBH[:, 1], color='black')
        plt.plot(Ret[1], Ret[2])
        plt.show()
    if nDilation:
        return BHs.GRangle(Ret[:, -1])

def binaryIntegration(N, MBH, PosBH, R, ang, v, FullOutput=True, method=1): # Integrate the momentum transfer over impact parameters.
    BHs = BH_Orbit(MBH, PosBH, v, method=method)
    T = 2e5
    dlam = 0.05
    Ni = max(40, int(N//4))

    XY = []
    # R = 400
    RR = 1e4
    B1 = np.linspace(-RR, -R, Ni, endpoint=False)
    B = np.linspace(-R, R, Ni, endpoint=False)
    B2 = np.linspace(R, RR, Ni, endpoint=True)
    # print(corr)
    for b in B1:
        zzz = BHs.SimMomentum(b, T, dlam, ang)
        XY +=[[b, zzz[0], zzz[1], zzz[2], zzz[3]]]
    for b in B:
        zzz = BHs.SimMomentum(b, T, dlam, ang)
        XY +=[[b, zzz[0], zzz[1], zzz[2], zzz[3]]]
    for b in B2:
        zzz = BHs.SimMomentum(b, T, dlam, ang)
        XY +=[[b, zzz[0], zzz[1], zzz[2], zzz[3]]]
    Counter = N - Ni
    while Counter > 0:
        Z = []
        for i in range(1, len(XY)):
            z = abs(((XY[i][1] - XY[i][2]) - (XY[i-1][1] - XY[i-1][2])))*(XY[i][0] - XY[i-1][0])
            if z> 100:
                Z.append(0)
            else:
                Z.append(z)

        m = np.argmax(np.array(Z)) +1
        x0 = (XY[m-1][0]+XY[m][0])/2
        y0 = BHs.SimMomentum(x0, T, dlam, ang)
        XY = XY[:m] + [[x0, y0[0], y0[1], y0[2], y0[3]]] + XY[m:]

        W = []
        for i in range(1, len(XY)-1):
            x = [XY[i-1][0], XY[i][0], XY[i+1][0]]
            y = [XY[i-1][1]-XY[i-1][2], XY[i][1]-XY[i][2], XY[i+1][1]-XY[i+1][2]]
            w = (x[2]-x[0])*abs(y[0]+ (y[2]-y[0])*(x[1]-x[0])/(x[2] - x[0]))
            if w > 100:
                W.append(0)
            else:
                W.append(w)

        m = np.argmax(np.array(W)) +1

        x1 = (XY[m-1][0]+XY[m][0])/2
        x2 = (XY[m][0]+XY[m+1][0])/2
        y1 = BHs.SimMomentum(x1, T, dlam, ang)
        y2 = BHs.SimMomentum(x2, T, dlam, ang)
        XY = XY[:m] + [[x1, y1[0], y1[1], y1[2], y1[3]]] + [XY[m]] + [[x2, y2[0], y2[1], y2[2], y2[3]]] + XY[m+1:]
        Counter -=3

    if FullOutput:
        return np.array(XY)
    else:
        XY = np.array(XY)
        return np.array([scipy.integrate.trapezoid(XY[:, 1], XY[:, 0]), 
                         scipy.integrate.trapezoid(XY[:, 2], XY[:, 0]), 
                         scipy.integrate.trapezoid(XY[:, 3], XY[:, 0]), 
                         scipy.integrate.trapezoid(XY[:, 4], XY[:, 0])])


def angIntegration(bN, angN, R, MBH, PosBH, v, method=1):
    P = np.zeros((angN, 5))
    Ang = np.linspace(0, 2*np.pi, angN, endpoint=False)
    for i, ang in enumerate(Ang):
        p1, p2, p3, p4 = binaryIntegration(bN, MBH, PosBH, R, ang, v, FullOutput=False, method=method)
        print(i, p1, p2, p3, p4)
        P[i] = [ang, p1, p2, p3, p4]
    return P


def plotBinaryIntegration(N, MBH, PosBH, R, ang, v, method=1):
    P = binaryIntegration(N, MBH, PosBH, R, ang, v, True, method=method)

    sep = norm(PosBH[0] - PosBH[1])
    F = [scipy.integrate.trapezoid(P[:, i], P[:, 0]) for i in range(1, 5)]

    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 12}

    plt.rc('font', **font)
    fig, ax = plt.subplots(1, figsize=(10, 9), tight_layout=True)

    ax.scatter(P[:, 0], P[:, 1], label=f'BH1_x, total force: {np.round(F[0], 6)}')
    ax.scatter(P[:, 0], P[:, 2], label=f'BH2_x, total force: {np.round(F[1], 6)}, diSimMomentumerence {np.round(F[0] - F[1], 6)}')
    ax.scatter(P[:, 0], P[:, 3], label=f'BH1_y, total force: {np.round(F[2], 6)}')
    ax.scatter(P[:, 0], P[:, 4], label=f'BH2_y, total force: {np.round(F[3], 6)}, diSimMomentumerence {np.round(F[2] - F[3], 6)}')

    ax.set_xlim(-10*sep, 10*sep)
    ax.set_xlabel('Impact parameter $D$')
    ax.set_ylabel('Momentum transfer $p$')
    ax.set_title(f'Momentum transfer from a photon flux to black hole versus impact parameter \n for a black hole separation of {np.round(sep, 6)} and incoming angle {np.round(ang, 6)}')
    ax.legend()
    if method == 1:
        Methodstr = 'Rutherford'
    else: 
        Methodstr = 'Extremal'
    np.save(f'{Methodstr}ForceOneAngle{ang}_bN{N}_MBH_{MBH[0]}_a{PosBH[0, 0]}_v{v}', P, allow_pickle=True)
    plt.savefig(f'{Methodstr}ForceOneAngle{ang}_bN{N}_MBH_{MBH[0]}_a{PosBH[0, 0]}_v{v}.pdf', bbox_inches='tight')
    plt.show()




def plotAngIntegration(bN, angN, R, MBH, PosBH, v, method=1):
    P = angIntegration(bN, angN, R, MBH, PosBH, v, method)
    sep = norm(PosBH[0] - PosBH[1])
    # P = np.append(P, P[0])
    print(P)
    F = [scipy.integrate.trapezoid(np.append(P[:, i], P[0, i]), np.append(P[:, 0], 2*np.pi)) for i in range(1, 5)]
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 12}

    plt.rc('font', **font)

    fig, ax = plt.subplots(1, figsize=(10, 9), tight_layout=True)

    ax.set_xlabel('incoming angle of photons in radians')
    ax.set_ylabel('momentum transfer $p$')
    ax.set_title(f'Momentum transfer from a matter flux to black hole versus angle \n for a black hole separation of {sep}')
    ax.plot(np.append(P[:, 0], 2*np.pi), np.append(P[:, 1], P[0, 1]), label=f'BH 1_x, total force: {F[0]}')
    ax.plot(np.append(P[:, 0], 2*np.pi), np.append(P[:, 2], P[0, 2]), label=f'BH 2_x, total force: {F[1]}')
    ax.plot(np.append(P[:, 0], 2*np.pi), np.append(P[:, 3], P[0, 3]), label=f'BH 1_y, total force: {F[2]}')
    ax.plot(np.append(P[:, 0], 2*np.pi), np.append(P[:, 4], P[0, 4]), label=f'BH 2_y, total force: {F[3]}')
    ax.legend()
    if method == 1:
        Methodstr = 'Rutherford'
    else: 
        Methodstr = 'Extremal'
    np.save(f'{Methodstr}Force_bN{bN}_angN_{angN}_MBH_{MBH[0]}_a{PosBH[0, 0]}_v{v}.pdf', P, allow_pickle=True)
    plt.savefig(f'{Methodstr}Force_bN{bN}_angN_{angN}_MBH_{MBH[0]}_a{PosBH[0, 0]}_v{v}.pdf', bbox_inches='tight')
    plt.show()


t0 = datetime.now()

MBH = [1, 1]
NBH= len(MBH)

PosBH = np.array([[-2000, 0], 
                  [2000, 0]])


# setupPlot(5000, 623, 2000, 0, 0.05, MBH, PosBH, method=3)

# plotBinaryIntegration(300, MBH, PosBH, 1000, np.pi/2, 0, method=3)

plotAngIntegration(500, 20, 2000, MBH, PosBH, 0.0, method=3)



