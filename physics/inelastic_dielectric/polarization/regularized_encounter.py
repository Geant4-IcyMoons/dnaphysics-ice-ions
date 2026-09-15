"""Planar Levi-Civita fallback for a failed direct encounter.

For r''=-mu*r/|r|^3+F(r,t), set r=L(u)u and dt/ds=|u|^2.
Then u''=h*u/2+|u|^2*L(u).T*F/2 and h'=2*u'.L(u).T*F,
where h=|dr/dt|^2/2-mu/|r|. This keeps the point Coulomb force.
The perturbation here is the moving harmonic trap plus the frozen cloud.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc
from physics.inelastic_dielectric.polarization.nonlinear_oscillator import _StableDOP853


def lc_position(u):
    return np.array([u[0]*u[0]-u[1]*u[1], 2*u[0]*u[1]])


def lc_matrix(u):
    return np.array([[u[0], -u[1]], [u[1], u[0]]])


def initial_lc(r, p, mu):
    u = np.array([np.sqrt((np.linalg.norm(r)+r[0])/2), 0.])
    if u[0] != 0:
        u[1] = r[1]/(2*u[0])
    else:
        u[1] = np.sqrt(np.linalg.norm(r))
    return u, .5*lc_matrix(u).T@p, float(.5*np.dot(p,p)-mu/np.linalg.norm(r))


def enclosed(field, radius):
    density = getattr(field, 'density', None)
    if density is None or density.electrons == 0:
        return 0.
    if density.electrons == 1:
        return float(gammainc(3, 2*density.z*radius))
    return float(sum(weight*gammainc(l+1.5, a*radius*radius) for l,a,weight in density.terms))


def encounter(x, b, eta, field, *, gamma=1., tail=64., rtol=2e-9,
              max_fictitious_time=10000., max_evaluations=1000000):
    if (not np.all(np.isfinite([x,b,eta,gamma,tail,rtol,max_fictitious_time]))
            or min(x,b,eta,tail,max_fictitious_time) <= 0 or gamma < 1 or not 0 < rtol < 1):
        raise ValueError('Invalid regularized encounter inputs')
    end = tail*max(4., 1/x)
    mu = eta*field.z
    r0 = np.array([1., -end]); p0 = np.array([0., 1.])
    u, up, h = initial_lc(r0, p0, mu)
    initial = np.r_[u, up, h, -end, np.zeros(4)]
    evaluations = 0

    def rhs(s, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise RuntimeError('Regularized encounter exhausted evaluation budget')
        u, up, h, tau = state[:2], state[2:4], state[4], state[5]
        rho = float(np.dot(u,u)); r = lc_position(u); L = lc_matrix(u)
        # N_inside ~ r^3 makes the screening perturbation regular at r=0.
        screen = 0. if rho == 0 else eta*enclosed(field,b*rho)/rho**3
        F = x*x*(np.array([1.,tau])-r)+screen*r
        lforce = L.T@F
        R = np.array([1.,tau]); norm = np.linalg.norm(R)
        leading_force = field.scalar(b*norm)*R/norm**3
        return np.r_[up, .5*h*u+.5*rho*lforce, 2*np.dot(up,lforce), rho,
                     rho*leading_force*np.cos(x*tau), rho*leading_force*np.sin(x*tau)]

    def finish(s, state):
        return state[5]-end
    finish.terminal = True
    finish.direction = 1
    solved = solve_ivp(rhs, (0.,max_fictitious_time), initial, method=_StableDOP853,
                       rtol=rtol, atol=rtol*.01, max_step=.05, events=finish)
    if not solved.success or not len(solved.t_events[0]):
        raise RuntimeError(f'Regularized encounter did not reach final physical time: {solved.message}; tau={solved.y[5,-1]}')
    u, up = solved.y[:2,-1],solved.y[2:4,-1]
    rho = float(np.dot(u,u)); r = lc_position(u); p = 2*lc_matrix(u)@up/rho
    tau = solved.y[5,-1]
    y = (np.array([1.,tau])-r)/eta
    yp = (np.array([0.,1.])-p)/eta
    c0,s0 = solved.y[6:8,-1],solved.y[8:10,-1]
    weights = np.array([1.,gamma**-2])
    leading = .5*np.dot(weights,c0*c0+s0*s0)
    full = .5*np.dot(weights,yp*yp+x*x*y*y)
    radii = np.sum(solved.y[:2]**2,axis=0)
    return dict(full=float(full),leading=float(leading),difference=float(full-leading),
                evaluations=evaluations,minimum_accepted_separation_bohr=float(b*radii.min()),
                final_separation_bohr=float(b*rho), final_tau=float(tau),
                kepler_energy_residual=float(solved.y[4,-1]-(.5*np.dot(p,p)-mu/rho)))
