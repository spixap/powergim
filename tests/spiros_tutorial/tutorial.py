# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:59:01 2022

Just a custom tutorial to get familiar with powergim


@author: spyridonc
"""
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import powergim as pgim


print(Path(__file__).parents[1]) # this works only if i run the fule with run button. Otherwise i do not specify the module so that Path can find the path of the __file__ variable

# Define location of INPUT files
TEST_DATA_ROOT_PATH = Path(__file__).parents[1] / "test_data"
 
NUMERIC_THRESHOLD = 1e-3 # FIXME: specify what is this

# ------------------------Read INPUT data (costs)------------------------------
# 1. Cost and other economical parameters 
parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters.yaml")

# 2. Grid Layout (available nodes, branches, generators, consumers )
grid_data      = pgim.file_io.read_grid(
      investment_years=parameter_data["parameters"]["investment_years"],
      nodes=TEST_DATA_ROOT_PATH / "nodes.csv",
      branches=TEST_DATA_ROOT_PATH / "branches.csv",
      generators=TEST_DATA_ROOT_PATH / "generators.csv",
      consumers=TEST_DATA_ROOT_PATH / "consumers.csv",
  )

# 3. Profiles (timeseries)
file_timeseries_sample = TEST_DATA_ROOT_PATH / "time_series_sample.csv"
grid_data.profiles     = pgim.file_io.read_profiles(filename=file_timeseries_sample)

grid_data.branch.loc[:, "max_newCap"] = 5000 # TODO: why setting max_newCap?

# --------------------------Create powerGIM model------------------------------
model = pgim.SipModel(grid_data=grid_data, parameter_data=parameter_data)
grid_data.branch["dist_computed"] = grid_data.compute_branch_distances()

