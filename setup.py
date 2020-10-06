# Copyright (c) 2019-2020  Elias Fernandez
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

from setuptools import setup

setup(
    name='EGTtools',
    version='0.0.1',
    packages=['egttools', 'egttools.tests', 'egttools.analytical', 'egttools.plotting'],
    url='',
    license='MIT License',
    author='Elias Fernandez',
    author_email='elias.fernandez.domingos@vub.be',
    description='Computational tools for studying Game Theoretical problems '
                'using the Evolutionary Game Theory Framework.'
)
