"""Direct, unsmoothed NLH/ZBL encounter partition for handoff experiments."""
from functools import lru_cache
import math
import numpy as np
from scipy.optimize import brentq
from physics.elastic.bca.scattering import NLHCollisionKernel, two_body_outcome_from_cm_angle, _gauss_legendre
from physics.elastic.bca.runtime import RuntimeCollision, KernelTableError
from physics.elastic.zbl.backend import FullZBLKernel
from physics.elastic.zbl.kernel import screening_length_angstrom, COULOMB_EV_ANGSTROM


class HandoffKernel(FullZBLKernel):
    """One impact disk, one branch per encounter, shared isotope kinematics.

    Direct orbit integration is the reference path; no unvalidated interpolation
    is introduced. boundary_ev=None gives the matched full-ZBL control.
    """
    def __init__(self, *, minimum_transfer_ev, boundary_ev=30., order=64):
        if boundary_ev is not None and (not math.isfinite(boundary_ev) or boundary_ev < 30):
            raise ValueError('Boundary must be >=30 eV or None for full ZBL.')
        if order < 16:
            raise ValueError('Orbit order must be >=16.')
        super().__init__(minimum_transfer_ev=minimum_transfer_ev, projectiles=('C',), energy_bounds_ev=(1.,1e8))
        self.boundary_ev, self.order = boundary_ev, order

    @lru_cache(maxsize=2048)
    def hard_kernel(self, target, energy):
        return NLHCollisionKernel('C',target,energy,minimum_turning_potential_ev=self.boundary_ev or 30.,quadrature_order=self.order)

    def boundary(self, target, energy):
        return 0. if self.boundary_ev is None else self.hard_kernel(target,energy).maximum_impact_parameter_angstrom

    def branch(self, target, energy, impact):
        return 'nlh' if self.boundary_ev is not None and impact <= self.boundary(target,energy) and self.boundary(target,energy)>0 else 'zbl'

    def zbl_angle(self, target, energy, impact):
        if impact == 0: return math.pi
        context=self.pair_kinematics('C',target,energy)
        z=1 if target=='H' else 8
        length=screening_length_angstrom(6,z)
        def potential(r):
            x=np.asarray(r)/length
            return COULOMB_EV_ANGSTROM*6*z/np.asarray(r)*sum(a*np.exp(-b*x) for a,b in ((.1818,3.2),(.5099,.9423),(.2802,.4029),(.02817,.2016)))
        rel=context.relative_kinetic_energy_ev
        def f(r):return 1-(impact/r)**2-float(potential(r))/rel
        high=max(1.,2*impact)
        while f(high)<0: high*=2
        radius=brentq(f,impact,high,xtol=1e-13,rtol=1e-14)
        angles,weights=_gauss_legendre(self.order)
        cosine=np.cos(angles)
        rad=1-(impact*cosine/radius)**2-potential(radius/cosine)/rel
        if np.any(rad<=0): raise KernelTableError('Invalid orbit radicand.')
        half=impact/radius*np.sum(weights*np.sin(angles)/np.sqrt(rad))
        return float(np.clip(math.pi-2*half,0,math.pi))

    @lru_cache(maxsize=4096)
    def maximum_impact_parameter_angstrom(self, projectile, target, projectile_energy_ev):
        c=self.pair_kinematics(projectile,target,projectile_energy_ev)
        maximum=two_body_outcome_from_cm_angle(c,math.pi).recoil_energy_ev
        if maximum<=self.minimum_transfer_ev:
            outer=0.
        else:
            def f(b):return two_body_outcome_from_cm_angle(c,self.zbl_angle(target,projectile_energy_ev,b)).recoil_energy_ev-self.minimum_transfer_ev
            hi=1.
            while f(hi)>0: hi*=2
            outer=brentq(f,0.,hi,xtol=1e-10)
        # Keep the entire NLH disk even if the soft annulus is empty.
        return max(outer,self.boundary(target,projectile_energy_ev))

    def theta_cm_rad(self, projectile,target,projectile_energy_ev,area_quantile):
        impact=self.impact_parameter_from_area_quantile(projectile,target,projectile_energy_ev,area_quantile)
        return self.collide(projectile,target,projectile_energy_ev,impact).theta_cm_rad

    def area_quantile_breakpoints(self,projectile,target,projectile_energy_ev):
        outer=self.maximum_impact_parameter_angstrom(projectile,target,projectile_energy_ev)
        b=self.boundary(target,projectile_energy_ev)
        return np.unique([0.,(b/outer)**2 if outer else 0.,1.])

    def collide(self,projectile,target,projectile_energy_ev,impact_parameter_angstrom):
        c=self.pair_kinematics(projectile,target,projectile_energy_ev)
        b=float(impact_parameter_angstrom)
        outer=self.maximum_impact_parameter_angstrom(projectile,target,projectile_energy_ev)
        if not math.isfinite(b) or b<0 or b>outer*(1+1e-12): raise KernelTableError('Impact outside retained disk.')
        theta=(self.hard_kernel(target,projectile_energy_ev).solve(b).theta_cm_rad
               if self.branch(target,projectile_energy_ev,b)=='nlh' else self.zbl_angle(target,projectile_energy_ev,b))
        o=two_body_outcome_from_cm_angle(c,theta)
        return RuntimeCollision(projectile,target,projectile_energy_ev,c.relative_kinetic_energy_ev,b,outer,theta,o.theta_projectile_lab_rad,o.recoil_energy_ev,o.projectile_out_energy_ev,o.energy_conservation_error_ev)
