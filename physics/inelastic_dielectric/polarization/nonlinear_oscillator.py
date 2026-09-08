"""Full frozen-projectile oscillator response and its impact integral.

The production correction is full minus leading response, not a cubic
coefficient. See NONLINEAR_POLARIZATION.md for the energy conversion and
the explicitly approximate extension of the previous gamma prescription.
No cubic or point-charge analytic fallback is provided.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.integrate import DOP853, cumulative_simpson, quad, quad_vec, solve_ivp
from scipy.interpolate import CubicSpline

from physics.inelastic_dielectric.projectile_potentials.projectile_form_factors import load_density

VERSION = "full-minus-leading-rotating-amplitude-ode-v1"
RELATIVE_TOLERANCE = 1e-3
BENCHMARK_RTOL = 1e-3
MAX_X = 50.


class _StableDOP853(DOP853):
    """SciPy's embedded error norm with scaling before squaring tiny tails."""

    def _estimate_error_norm(self, K, h, scale):
        error5 = K.T @ self.E5 / scale
        error3 = K.T @ self.E3 / scale
        peak = max(np.max(np.abs(error5)), np.max(np.abs(error3)))
        if peak == 0:
            return 0.
        if not np.isfinite(peak):
            return np.inf
        norm5 = np.sum((error5 / peak)**2)
        norm3 = np.sum((error3 / peak)**2)
        return abs(h) * peak * norm5 / np.sqrt((norm5 + .01*norm3) * len(scale))


class RadialField:
    """Gauss-law field from a frozen density, with a positive tail spline."""

    def __init__(self, density, nodes=8193):
        self.density = density
        self.z = density.z
        self.radius = np.sqrt(density.moment(2)/density.electrons) if density.electrons else 1.
        self.electron_integral = 0.
        if not density.electrons:
            return
        upper = 100*self.radius
        if density.electrons > 1:
            upper = max(upper, np.sqrt(1500/np.min(density.terms[:, 1])))
        r = np.geomspace(1e-7*self.radius, upper, nodes)
        lr = np.log(r)
        integrand = 4*np.pi*r**3*density.density(r)
        if np.any(~np.isfinite(integrand)) or np.min(integrand) < -1e-12:
            raise RuntimeError("Nonphysical frozen density")
        tail = cumulative_simpson(integrand[::-1], x=-lr[::-1], initial=0.)[::-1]
        self.electron_integral = float(tail[0])
        if not np.isclose(self.electron_integral, density.electrons, rtol=1e-9, atol=1e-10):
            raise RuntimeError("Frozen field fails electron normalization")
        keep = tail > 1e-260
        self.lower, self.upper = r[0], r[keep][-1]
        self.spline = CubicSpline(lr[keep], np.log(tail[keep]), extrapolate=False)

    def scalar(self, r):
        d = self.density
        if not d.electrons or r < self.lower:
            return float(d.z)
        if r > self.upper:
            return float(d.charge)
        return float(d.charge+np.exp(self.spline(np.log(r))))

    def __call__(self, r):
        r = np.asarray(r, float)
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("Radius must be finite and nonnegative")
        return np.array([self.scalar(v) for v in r.flat]).reshape(r.shape)


class PointField:
    """Point interaction charge, including the existing scalar charge modes."""
    def __init__(self, charge):
        if not np.isfinite(charge) or charge < 0:
            raise ValueError("Point interaction charge must be finite and nonnegative")
        self.z = float(charge)

    def scalar(self, radius):
        return self.z


@lru_cache(maxsize=80)
def frozen_field(element, charge, nodes=8193):
    return RadialField(load_density(element, charge), nodes)


def encounter(x, b, eta, field, *, gamma=1., tail=64., rtol=2e-11):
    """Scaled oscillator energies and stable full-minus-leading difference.

    y0''+x^2*y0=f(R), d''+x^2*d=f(R-eta*(y0+d))-f(R).
    Evolve rotating harmonic amplitudes, y=(C*sin(x*tau)-S*cos(x*tau))/x,
    C'=f*cos(x*tau), S'=f*sin(x*tau), and their full-minus-leading
    differences. Free oscillations are then exact, rather than followed
    numerically through the long tails. With tau=gamma*v*t/b, displacements are
    (b*eta*y_x, b*eta*y_z/gamma). Thus H is weighted by (1,gamma^-2),
    and physical energy is E_h*gamma^2*v^2*eta^2*H.
    This is the existing electric-field kinematic prescription extended
    without truncation, not a full relativistic electron/Lorentz-force model.
    """
    if (not np.all(np.isfinite([x,b,eta,gamma,tail,rtol])) or x <= 0 or b <= 0
            or gamma < 1 or tail <= 0 or not 0 < rtol < 1):
        raise ValueError("Invalid nonlinear oscillator input")
    end = np.arcsinh(tail*max(4., 1/x))
    evaluations = 0

    def rhs(s, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 500000:
            raise RuntimeError("Nonlinear encounter exceeded the ODE evaluation budget")
        tau = math.sinh(s)
        cs,sn=math.cos(x*tau),math.sin(x*tau)
        cx,cz,sx,sz,dcx,dcz,dsx,dsz=state
        rx=1-eta*(sn*(cx+dcx)-cs*(sx+dsx))/x
        rz=tau-eta*(sn*(cz+dcz)-cs*(sz+dsz))/x
        distance=math.hypot(rx,rz)
        if distance < 1e-8:
            raise RuntimeError("Trajectory approaches the nuclear singularity; no softened force was substituted")
        h=math.hypot(1.,tau)
        f0x=field.scalar(b*h)/h**3
        f0z=f0x*tau
        force=field.scalar(b*distance)/distance**3
        dfx,dfz=force*rx-f0x,force*rz-f0z
        c,sn=math.cosh(s)*cs,math.cosh(s)*sn
        return [c*f0x,c*f0z,sn*f0x,sn*f0z,c*dfx,c*dfz,sn*dfx,sn*dfz]

    solved = solve_ivp(rhs, (-end,end), np.zeros(8), method=_StableDOP853,
                       rtol=rtol, atol=rtol*.01, max_step=.1, t_eval=[end])
    if not solved.success or np.any(~np.isfinite(solved.y)):
        raise RuntimeError("Nonlinear oscillator failed: "+solved.message)
    c0,s0,dc,ds = np.split(solved.y[:,-1],4)
    weights = np.array([1.,gamma**-2])
    leading = .5*float(weights @ (c0*c0+s0*s0))
    full = .5*float(weights @ ((c0+dc)**2+(s0+ds)**2))
    difference = float(weights @ (c0*dc+.5*dc*dc+s0*ds+.5*ds*ds))
    return dict(leading=leading, full=full, difference=difference, evaluations=evaluations)


def nonlinear_energy(x, b, eta, field, *, tail=64., rtol=2e-11):
    """Nonrelativistic H(eta) for the retained encounter comparison."""
    return encounter(x,b,eta,field,tail=tail,rtol=rtol)["full"]


def linear_energy_quadrature(x, b, field, *, gamma=1.):
    """Independent infinite-time Fourier integral of the undisplaced force."""
    def force(tau, component):
        h=np.hypot(1.,tau)
        return field.scalar(b*h)/h**3*(tau if component else 1.)
    px=2*quad(force,0,np.inf,args=(0,),weight="cos",wvar=x,epsabs=1e-11,limlst=150)[0]
    pz=2*quad(force,0,np.inf,args=(1,),weight="sin",wvar=x,epsabs=1e-11,limlst=150)[0]
    return (px*px+pz*pz/gamma**2)/2


def integrate_kernel(xi, b_per_x, gamma, field, *, rtol=RELATIVE_TOLERANCE,
                     atol=None, allow_unconverged=False):
    """K = integral [H_full-H_leading]/(2*eta*x) d(log x).

    eta=1/(gamma*b*v^2), v=0.5616*C_B/(xi*b_per_x); C_B=1.
    This normalization retains the previous cm2/eV prefactor, but K is no
    longer cubic or necessarily positive. The zero-charge response is exact.
    Time/tolerance errors are integrated as absolute values, separately from
    impact quadrature. No Born-relative error floor or cubic fallback is used.
    """
    if atol is None:
        atol=1e-9*max(1.,field.z**3)
    if (not np.all(np.isfinite([xi,b_per_x,gamma,rtol,atol])) or xi <= 0 or b_per_x <= 0
            or gamma < 1 or not 0 < rtol < 1 or atol <= 0):
        raise ValueError("Invalid nonlinear impact integral")
    velocity=.5616/(xi*b_per_x)
    if xi >= MAX_X or field.z == 0:
        return 0.,0.,dict(converged=True,impact_error=0.,time_error=0.,tail_error=0.,evaluations=0)
    points=[np.log(p) for p in (.01,.1,1.,10.) if xi < p < MAX_X]
    if hasattr(field,"radius"):
        transition=field.radius/b_per_x
        if xi < transition < MAX_X:
            points.append(np.log(transition))
    evaluations=0
    for level in range(3):
        tail=16.*2**level
        ode_rtol=2e-8/10**level

        def integrand(logx):
            nonlocal evaluations
            x=np.exp(logx)
            b=x*b_per_x
            eta=1/(gamma*b*velocity**2)
            window=tail*max(1.,1/x)/max(4.,1/x)
            low=encounter(x,b,eta,field,gamma=gamma,tail=window,rtol=ode_rtol)
            fine=encounter(x,b,eta,field,gamma=gamma,tail=window,rtol=ode_rtol/10)
            extended=encounter(x,b,eta,field,gamma=gamma,tail=2*window,rtol=ode_rtol/10)
            evaluations+=1
            weight=1/(2*eta*x)
            return weight*np.array([extended["difference"],abs(fine["difference"]-low["difference"]),
                                    abs(extended["difference"]-fine["difference"])])

        values,impact_error,info=quad_vec(integrand,np.log(xi),np.log(MAX_X),
            points=sorted(set(points)),epsabs=atol/8,epsrel=rtol/8,norm="max",limit=128,
            quadrature="gk21",full_output=True)
        value,time_error,tail_error=values
        error=float(impact_error+time_error+tail_error)
        converged=bool(info.success and np.isfinite(value) and np.isfinite(error)
                       and error <= atol+rtol*abs(value))
        diagnostics=dict(converged=converged,impact_error=float(impact_error),time_error=float(time_error),
                         tail_error=float(tail_error),evaluations=evaluations,refinement=level)
        if converged:
            return float(value),error,diagnostics
    if allow_unconverged:
        return float(value),error,diagnostics
    raise RuntimeError(f"Nonlinear polarization integral did not converge: xi={xi:g}, K={value:g}, {diagnostics}")
