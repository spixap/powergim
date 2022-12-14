# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:46:29 2022

Tutorial on PowerGIM solving a stochastic model using the extensive form

@author: spyridonc
"""

from pathlib import Path

import mpi4py
import mpisppy.opt.ph
import mpisppy.utils.sputils

import pandas as pd
import pyomo.environ as pyo
import powergim as pgim



# Define location of INPUT files
TEST_DATA_ROOT_PATH = Path(__file__).parents[1] / "stochastic" / "data"



NUM_SCENARIOS = 4
TMP_PATH = Path()

 
# ------------------------Read INPUT data (costs)------------------------------
# 1. Cost and other economical parameters 
parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters_stoch.yaml")


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

grid_data.branch.loc[:, "max_newCap"] = 5000 # Set temporarily to reproduce previous result

# 4. Define 3 scenarios
scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
scenario_names = ["scen0", "scen1", "scen2"] # TODO: try to define one addtional scenario

# -----------------------Create scenario powerGIM models-----------------------
# This function edits the inital grid_data and parameters depending on the scenario
# Then it creates a model instance for the scenario with the modified parameters
def my_scenario_creator(scenario_name, grid_data, parameter_data):
    """Create a scenario."""
    print("Scenario {}".format(scenario_name))

    # Adjust data according to scenario name
    # FIXME: probabilities should be created only once
    num_scenarios = NUM_SCENARIOS
    probabilities = {f"scen{k}": 1 / num_scenarios for k in range(num_scenarios)}
    # probabilities = {"scen0": 0.3334, "scen1": 0.3333, "scen2": 0.3333,"scen"}
    
    if scenario_name == "scen0":
        pass
    elif scenario_name == "scen1":
        # Less wind at SN2
        grid_data.generator.loc[4, "capacity_2028"] = 1400
    elif scenario_name == "scen2":
        # More wind and SN2
        grid_data.generator.loc[4, "capacity_2028"] = 10000
        grid_data.generator.loc[3, "capacity_2028"] = 10000
    elif scenario_name == "scen3":
        # More wind, more demand
        grid_data.generator.loc[4, "capacity_2028"] = 8000
    elif scenario_name == "scen4":
        grid_data.generator.loc[4, "capacity_2028"] = 9000
    elif scenario_name == "scen5":
        grid_data.generator.loc[4, "capacity_2028"] = 10000
    else:
        raise ValueError("Invalid scenario name")

    # Create stochastic model:
    # A) Initialize a pgim object instane (gimModel)
    gimModel = pgim.SipModel(grid_data, parameter_data)
    # B) Use scenario_creator method to build a scenario instance model
    gimScenarioModel = gimModel.scenario_creator(scenario_name, probability=probabilities[scenario_name])
    return gimScenarioModel

# This formulates the extensive form based on the scenarios that are defined (an mpisppy object)
# Preferred method based on mpispppy: mpisppy.opt.ef.ExtensiveForm --> needs mpi4py
gimModel_ef = mpisppy.utils.sputils.create_EF(
    scenario_names,
    scenario_creator=my_scenario_creator,
    scenario_creator_kwargs=scenario_creator_kwargs,
)

# Solve the EF
solver = pyo.SolverFactory("glpk")
solver.solve(
    gimModel_ef,
    tee=True,
    symbolic_solver_labels=True,
)

all_var_all_scen_values = []

# Extract results:
for scen in mpisppy.utils.sputils.ef_scenarios(gimModel_ef):
    # Iterable has 2 dimensions: (scenario_name, scnenario_model (associated pyomo model variables))
    scen_name = scen[0]
    this_scen = scen[1]
    all_var_values = pgim.SipModel.extract_all_variable_values(this_scen)
    all_var_all_scen_values.append(all_var_values)
    print(f"{scen_name}: OBJ = {pyo.value(this_scen.OBJ)}")
    print(f"{scen_name}: opCost = {all_var_values[f'{scen_name}.v_operating_cost'].values}")

    # sputils.ef_nonants_csv(main_ef, "sns_results_ef.csv")
    # sputils.ef_ROOT_nonants_npy_serializer(main_ef, "sns_root_nonants.npy")
    print(f"EF objective: {pyo.value(gimModel_ef.EF_Obj)}")

# assert all_var_values["scen2.opCost"][1] == pytest.approx(2.0442991e10)
# assert all_var_values["scen2.opCost"][2] == pytest.approx(5.3318421e10)
    



