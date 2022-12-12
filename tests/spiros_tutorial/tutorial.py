# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:59:01 2022

Just a custom tutorial to get familiar with powergim


@author: spyridonc
"""
from pathlib import Path

import logging


import mpi4py

import mpisppy.opt.ph
import mpisppy.utils.sputils

import pandas as pd
import pyomo.environ as pyo

import powergim as pgim



# Define location of INPUT files
IN_FILES_PATH = Path(__file__).parents[1] / "stochastic" / "data"

# Define location of OUTPUT files
OUT_FILES_PATH = Path(__file__).parent / "outputs"


NUM_SCENARIOS = 3 # 4  for EF, 3 for PH
TMP_PATH = Path()

 
# ------------------------Read INPUT data (costs)------------------------------
# 1. Cost and other economical parameters 
parameter_data = pgim.file_io.read_parameters(IN_FILES_PATH / "parameters_stoch.yaml")


# 2. Grid Layout (available nodes, branches, generators, consumers )
grid_data      = pgim.file_io.read_grid(
      investment_years=parameter_data["parameters"]["investment_years"],
      nodes=IN_FILES_PATH / "nodes.csv",
      branches=IN_FILES_PATH / "branches.csv",
      generators=IN_FILES_PATH / "generators.csv",
      consumers=IN_FILES_PATH / "consumers.csv",
  )

# 3. Profiles (timeseries)
file_timeseries_sample = IN_FILES_PATH / "time_series_sample.csv"
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

# This function gets all PH iterations results per scenario in a tidy way and writes them in csv files
def my_scenario_denouement(rank, scenario_name, scenario):
    print(f"DENOUEMENT scenario={scenario_name} OBJ={pyo.value(scenario.OBJ)}")
    all_var_values_dict = pgim.SipModel.extract_all_variable_values(scenario)
    dfs = []
    for varname, vardata in all_var_values_dict.items(): # varname=key, vardata=value
        if vardata is None:
            logging.warning(f"tutorial.py: Skipping variable with no data inside ({varname})")
            continue
        df = pd.DataFrame(vardata).reset_index()
        df.loc[:, "variable"] = varname
        dfs.append(df)
    pd.concat(dfs).to_csv(OUT_FILES_PATH / f"ph_res_ALL_{scenario_name}.csv")
    
    

# Formulate the scenarios-structure to be solved by PH algorithm
def my_solve_ph(my_scenario_creator, my_scenario_denouement, scenario_creator_kwargs, scenario_names):

    # # Read input data
    # parameter_data = pgim.file_io.read_parameters(TEST_DATA_ROOT_PATH / "parameters_stoch.yaml")
    # grid_data = pgim.file_io.read_grid(
    #     investment_years=parameter_data["parameters"]["investment_years"],
    #     nodes=TEST_DATA_ROOT_PATH / "nodes.csv",
    #     branches=TEST_DATA_ROOT_PATH / "branches.csv",
    #     generators=TEST_DATA_ROOT_PATH / "generators.csv",
    #     consumers=TEST_DATA_ROOT_PATH / "consumers.csv",
    # )
    # file_timeseries_sample = TEST_DATA_ROOT_PATH / "time_series_sample.csv"
    # grid_data.profiles = pgim.file_io.read_profiles(filename=file_timeseries_sample)

    # Scenarios
    # scenario_creator_kwargs = {"grid_data": grid_data, "parameter_data": parameter_data}
    # # scenario_names = ["scen0", "scen1", "scen2", "scen3", "scen4", "scen5"]
    # scenario_names = [f"scen{k}" for k in range(NUM_SCENARIOS)]

    # Solve via progressive hedging (PH)
    options = {
        "solvername": "glpk",
        "PHIterLimit": 5,
        "defaultPHrho": 10,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        # TODO: ASK why do we need these linear-related stuff?
        "linearize_proximal_terms": True,
        "proximal_linearization_tolerance ": 0.1,  # default =1e-1
        "initial_proximal_cut_count": 2,  # default = 2 
        "iter0_solver_options": {},  # {"mipgap": 0.01},  # dict(),
        "iterk_solver_options": {},  # {"mipgap": 0.005},  # dict(),
    }
    ph = mpisppy.opt.ph.PH(
        options,
        scenario_names,
        scenario_creator=my_scenario_creator,
        scenario_denouement=my_scenario_denouement,  # post-processing and reporting
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    # solve
    conv, obj, tbound = ph.ph_main() 
    
    """
    * conv: The convergence value (not easily interpretable).
    * obj: 
        If `finalize=True`, this is the expected, weighted objective value with the proximal term included.
        This value is not directly useful. If `finalize=False`, this value is `None`.
    * tbound:
        The "trivial bound", computed by solving the model with no
        nonanticipativity constraints (immediately after iter 0).
    """
                    
    #TODO: clarify why this is here
    rank = mpi4py.MPI.COMM_WORLD.Get_rank()


    # Extract results:
    res_ph = []
    variables = ph.gather_var_values_to_rank0()
    df_res = None
    if variables is not None:
        # this is true when rank is zero.
        for (scenario_name, variable_name) in variables:
            variable_value = variables[scenario_name, variable_name]
            res_ph.append({"scen": scenario_name, "var": variable_name, "value": variable_value})
        df_res = pd.DataFrame(data=res_ph)
        print(f"{rank}: Saving to file...ph_res_rank0.csv")
        df_res.to_csv(OUT_FILES_PATH / "ph_res_rank0.csv")
    return ph, df_res


ph, df_res = my_solve_ph(my_scenario_creator, my_scenario_denouement,scenario_creator_kwargs, scenario_names)

# assert ph is not None
# assert isinstance(df_res, pd.DataFrame)





# This formulates the extensive form based on the scenarios that are defined (an mpisppy object)
# Preferred method based on mpispppy: mpisppy.opt.ef.ExtensiveForm --> needs mpi4py

# gimModel_ef = mpisppy.utils.sputils.create_EF(
#     scenario_names,
#     scenario_creator=my_scenario_creator,
#     scenario_creator_kwargs=scenario_creator_kwargs,
# )

# # Solve the EF
# solver = pyo.SolverFactory("glpk")
# solver.solve(
#     gimModel_ef,
#     tee=True,
#     symbolic_solver_labels=True,
# )



# Extract results:
# all_var_all_scen_values = []
# for scen in mpisppy.utils.sputils.ef_scenarios(gimModel_ef):
#     # Iterable has 2 dimensions: (scenario_name, scnenario_model (associated pyomo model variables))
#     scen_name = scen[0]
#     this_scen = scen[1]
#     all_var_values = pgim.SipModel.extract_all_variable_values(this_scen)
#     all_var_all_scen_values.append(all_var_values)
#     print(f"{scen_name}: OBJ = {pyo.value(this_scen.OBJ)}")
#     print(f"{scen_name}: opCost = {all_var_values[f'{scen_name}.v_operating_cost'].values}")

#     # sputils.ef_nonants_csv(main_ef, "sns_results_ef.csv")
#     # sputils.ef_ROOT_nonants_npy_serializer(main_ef, "sns_root_nonants.npy")
#     print(f"EF objective: {pyo.value(gimModel_ef.EF_Obj)}")

# assert all_var_values["scen2.opCost"][1] == pytest.approx(2.0442991e10)
# assert all_var_values["scen2.opCost"][2] == pytest.approx(5.3318421e10)
    



