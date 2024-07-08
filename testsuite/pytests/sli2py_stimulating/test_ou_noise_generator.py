# -*- coding: utf-8 -*-
#
# test_noise_generator.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Tests parameter setting and statistical correctness for one application.
"""

import nest
import numpy as np
import pytest


@pytest.fixture
def prepare_kernel():
    nest.ResetKernel()
    nest.resolution = 0.1


def test_ou_noise_generator_set_parameters(prepare_kernel):
    params = {"mean": 210.0, "std": 60.0, "dt": 0.1}

    oung1 = nest.Create("ou_noise_generator")
    oung1.set(params)

    nest.SetDefaults("ou_noise_generator", params)

    oung2 = nest.Create("ou_noise_generator")
    assert oung1.get(params) == oung2.get(params)


def test_ou_noise_generator_incorrect_noise_dt(prepare_kernel):
    with pytest.raises(nest.kernel.NESTError, match="StepMultipleRequired"):
        nest.Create("ou_noise_generator", {"dt": 0.25})

def test_ou_noise_generator_(prepare_kernel):
    # run for resolution dt=0.1 project to iaf_psc_alpha.
    # create 100 repetitions of 1000ms simulations
    # collect membrane potential at end
    nest.rng_seed = 20

    #oung = nest.Create("ou_noise_generator")
    oung = nest.Create('ou_noise_generator', 1, {'mean':0.0, 'std': 60.0, 'tau':1., 'dt':1.0})
    neuron = nest.Create("iaf_psc_alpha")
    # we need to connect to a neuron otherwise the generator does not generate
    nest.Connect(oung, neuron)

    ###
    ###
    ###
    ###
    ###
    # DEFAULT TAU IS ZERO WHICH CAUSES NAN CURRENT
    ###
    ###
    ###
    #oung.set({"mean": 0.0, "std": 60.0, 'tau':10., "dt": 1.0})

    # no spiking, all parameters 1, 0 leak potential
    #neuron.set({"V_th": 1e10, "C_m": 1.0, "tau_m": 1.0, "E_L": 0.0})

    mm = nest.Create('multimeter', 1, {'record_from':['I']})
    nest.Connect(mm, oung, syn_spec={'weight': 1})
    nest.Simulate(1000000.0)

    ou_current = mm.get('events')['I']
    curr_mean = np.mean(ou_current)
    curr_var = np.var(ou_current)
    expected_curr_mean = oung.mean
    expected_cur_var = oung.std**2 / 2

    # change this to check if the expected variance is close to the actual one
    # require mean within 3 std dev, std dev within 0.2 std dev
    import matplotlib.pyplot as plt
    plt.plot(ou_current)
    plt.show()
    breakpoint()
    assert np.abs(curr_mean - expected_curr_mean) < 3 * oung.std
    assert np.abs(curr_var - expected_cur_var) < 0.2 * curr_var

    '''
    # require mean within 3 std dev, std dev within three std dev of std dev
    assert np.abs(vm_mean - expected_vm_mean) < 3 * expected_vm_std
    assert np.abs(vm_std - expected_vm_std) < 3 * expected_vm_std_std
    '''
