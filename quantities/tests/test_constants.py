import numpy as np

from .. import constants as pc
from .. import units as pq
from .common import TestCase


class TestConstants(TestCase):

    def assertRelativelyClose(self, q1, q2, rtol=1e-6):
        # CODATA values are rounded, so relations between them hold only
        # approximately
        ratio = (q1 / q2).simplified
        self.assertEqual(ratio.dimensionality, pq.dimensionless.dimensionality)
        self.assertLess(abs(ratio.magnitude - 1), rtol)

    def test_all_constants_simplify(self):
        names = [
            name for name in dir(pc)
            if isinstance(getattr(pc, name), pc.UnitConstant)
        ]
        self.assertGreater(len(names), 300)
        for name in names:
            with self.subTest(name=name):
                magnitude = getattr(pc, name).simplified.magnitude
                self.assertTrue(np.isfinite(magnitude))
                self.assertNotEqual(magnitude, 0)

    def test_units(self):
        expected = {
            'h': pq.J * pq.s,
            'hbar': pq.J * pq.s,
            'e': pq.C,
            'k': pq.J / pq.K,
            'N_A': 1 / pq.mol,
            'R': pq.J / (pq.mol * pq.K),
            'm_e': pq.kg,
            'm_p': pq.kg,
            'epsilon_0': pq.F / pq.m,
            'mu_0': pq.N / pq.A**2,
            'G': pq.m**3 / (pq.kg * pq.s**2),
            'alpha': pq.dimensionless,
        }
        for name, units in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(pc, name).simplified.dimensionality,
                    (1 * units).simplified.dimensionality,
                )

    def test_relations(self):
        # reduced Planck constant: ħ = h/2π
        self.assertRelativelyClose(pc.hbar, pc.h / (2 * np.pi))
        # molar gas constant: R = N_A k
        self.assertRelativelyClose(pc.R, pc.N_A * pc.k)
        # Faraday constant: F = N_A e
        self.assertRelativelyClose(pc.Faraday_constant, pc.N_A * pc.e)
        # speed of light: ε₀μ₀c² = 1
        self.assertRelativelyClose(pc.epsilon_0 * pc.mu_0 * pq.c**2, pq.dimensionless)
        # impedance of vacuum: Z₀ = μ₀c
        self.assertRelativelyClose(pc.Z_0, pc.mu_0 * pq.c)
        # fine-structure constant: α = e²/4πε₀ħc
        self.assertRelativelyClose(
            pc.alpha, pc.e**2 / (4 * np.pi * pc.epsilon_0 * pc.hbar * pq.c)
        )
        # Bohr radius: a₀ = ħ/mₑcα
        self.assertRelativelyClose(pc.a_0, pc.hbar / (pc.m_e * pq.c * pc.alpha))
        # Bohr magneton: μ_B = eħ/2mₑ
        self.assertRelativelyClose(pc.mu_B, pc.e * pc.hbar / (2 * pc.m_e))
        # Rydberg constant: R∞ = α²mₑc/2h
        self.assertRelativelyClose(
            pc.R_infinity, pc.alpha**2 * pc.m_e * pq.c / (2 * pc.h)
        )
        # Stefan–Boltzmann constant: σ = 2π⁵k⁴/15h³c²
        self.assertRelativelyClose(
            pc.sigma, 2 * np.pi**5 * pc.k**4 / (15 * pc.h**3 * pq.c**2)
        )
        # electron volt: 1 eV = e × 1 V
        self.assertRelativelyClose(pc.eV, pc.e * pq.V)
