#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import sys
import unittest

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc
from do_mpc._casadi_compat import castools


# Reference result written with casadi <= 3.7, i.e. a pickle that carries the
# old 'casadi.tools.structure3' module path.
LEGACY_PICKLE = './results/results_lotka_volterra.pkl'

# Symbols do-mpc reads off castools. Up to casadi 3.7 these leaked into
# casadi.tools via 'from casadi import *'; since 3.8 they live in casadi only.
CASADI_SYMBOLS = [
    'DM', 'MX', 'SX', 'Function', 'collocation_points', 'evalf', 'horzcat',
    'integrator', 'inv', 'jacobian', 'nlpsol', 'substitute', 'sum1', 'sum2',
    'tangent', 'vertcat',
]

# Symbols that are genuinely part of casadi.tools in every supported version.
CASADI_TOOLS_SYMBOLS = [
    'entry', 'indexf', 'struct_MX', 'struct_SX', 'struct_symMX', 'struct_symSX',
]


class TestCasadiCompat(unittest.TestCase):
    """Guards against the casadi 3.8 namespace changes.

    casadi 3.8.0 stopped re-exporting the casadi namespace from casadi.tools
    and renamed casadi.tools.structure3 to casadi.tools.structure.
    """

    def test_casadi_symbols_resolve(self):
        """Symbols that used to leak into casadi.tools are still reachable."""
        for name in CASADI_SYMBOLS:
            self.assertTrue(hasattr(castools, name),
                            'castools.{} does not resolve'.format(name))

    def test_casadi_tools_symbols_resolve(self):
        """Genuine casadi.tools symbols are still reachable."""
        for name in CASADI_TOOLS_SYMBOLS:
            self.assertTrue(hasattr(castools, name),
                            'castools.{} does not resolve'.format(name))

    def test_structure3_alias(self):
        """structure3 and structure refer to the same module on all versions."""
        self.assertIs(castools.structure3, castools.structure)
        self.assertTrue(hasattr(castools.structure3, 'DMStruct'))

    def test_structure3_importable(self):
        """The old module path stays importable, so old pickles keep working."""
        import casadi.tools.structure3
        self.assertIs(casadi.tools.structure3, castools.structure3)

    def test_load_legacy_results(self):
        """Result files written with casadi <= 3.7 still load.

        This is the regression that is invisible to the rest of the suite: the
        module was importable but every saved result raised ModuleNotFoundError.
        """
        # Make sure the fixture still is a pre-3.8 pickle, so that this test
        # cannot silently pass if the reference files are ever regenerated.
        with open(LEGACY_PICKLE, 'rb') as f:
            self.assertIn(b'casadi.tools.structure3', f.read(),
                          '{} is no longer a pre-casadi-3.8 pickle, the '
                          'regression it guards is no longer covered'
                          .format(LEGACY_PICKLE))

        results = do_mpc.data.load_results(LEGACY_PICKLE)
        self.assertIn('mpc', results)
        self.assertGreater(results['mpc']['_x'].shape[0], 0)


if __name__ == '__main__':
    unittest.main()
