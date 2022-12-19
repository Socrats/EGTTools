# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

import pytest

egttools = pytest.importorskip("egttools")


def test_calculate_nb_states():
    assert egttools.numerical.numerical.calculate_nb_states == egttools.calculate_nb_states


def test_calculate_state():
    assert egttools.numerical.numerical.calculate_state == egttools.calculate_state


def test_sample_simple():
    assert egttools.numerical.numerical.sample_simplex == egttools.sample_simplex


def test_abstract_game():
    assert egttools.numerical.numerical.games.AbstractGame == egttools.games.AbstractGame


def test_normal_form_game():
    assert egttools.numerical.numerical.games.NormalFormGame == egttools.games.NormalFormGame


def test_abstract_nfg_strategy():
    assert egttools.numerical.numerical.behaviors.NormalForm.AbstractNFGStrategy == egttools.behaviors.NormalForm.AbstractNFGStrategy
