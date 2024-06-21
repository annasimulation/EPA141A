# Import necessary libraries
import pandas as pd
import numpy as np
from ema_workbench import (Model, RealParameter, IntegerParameter, CategoricalParameter,
                           ScalarOutcome, perform_experiments, Samplers, Policy, Scenario)
from ema_workbench.em_framework import (SequentialEvaluator, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import (EpsNSGAII, Convergence)
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress)
from ema_workbench.analysis import parcoords
from dike_model_function import DikeNetwork
import os
import datetime
from problem_formulation import (
    sum_over,
    get_model_for_problem_formulation,
    )
    # Define the sum_over function


# Define the function to get the model for the given problem formulation
# def get_model_for_problem_formulation(problem_formulation_id):
#     function = DikeNetwork()
#     dike_model = Model("dikesnet", function=function)
#
#     Real_uncert = {"Bmax": [30, 350], "pfail": [0, 1]}  # m and [.]
#     cat_uncert_loc = {"Brate": (1.0, 1.5, 10)}
#     cat_uncert = {f"discount rate {n}": (1.5, 2.5, 3.5, 4.5) for n in function.planning_steps}
#     Int_uncert = {"A.0_ID flood wave shape": [0, 132]}
#
#     uncertainties = []
#
#     for uncert_name in cat_uncert.keys():
#         categories = cat_uncert[uncert_name]
#         uncertainties.append(CategoricalParameter(uncert_name, categories))
#
#     for uncert_name in Int_uncert.keys():
#         uncertainties.append(IntegerParameter(uncert_name, Int_uncert[uncert_name][0], Int_uncert[uncert_name][1]))
#
#     dike_height_levers = []
#     for dike in function.dikelist:
#         for uncert_name in Real_uncert.keys():
#             name = f"{dike}_{uncert_name}"
#             lower, upper = Real_uncert[uncert_name]
#             uncertainties.append(RealParameter(name, lower, upper))
#
#         for uncert_name in cat_uncert_loc.keys():
#             name = f"{dike}_{uncert_name}"
#             categories = cat_uncert_loc[uncert_name]
#             uncertainties.append(CategoricalParameter(name, categories))
#
#         for n in function.planning_steps:
#             name = f"{dike}_DikeIncrease {n}"
#             dike_height_levers.append(IntegerParameter(name, 0, 10))
#
#     dike_model.uncertainties = uncertainties
#
#     dike_height_levers.append(IntegerParameter("EWS_DaysToThreat", 0, 4))  # days
#     # Set levers: No RfR, dike heightening
#     dike_model.levers = dike_height_levers
#
#     # Define the outcomes
#     outcomes = [
#         ScalarOutcome('Total Costs', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
#             f"{dike}_Expected Annual Damage" for dike in function.dikelist] +
#                       [f"{dike}_Dike Investment Costs" for dike in function.dikelist] +
#                       ["RfR Total Costs"]
#         ),
#         ScalarOutcome('Expected Number of Deaths', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
#             f"{dike}_Expected Number of Deaths" for dike in function.dikelist]
#         ),
#         ScalarOutcome('Expected Annual Damage', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
#             f"{dike}_Expected Annual Damage" for dike in function.dikelist]),
#         ScalarOutcome('Dike Investment Costs', kind=ScalarOutcome.MINIMIZE, function=sum_over,
#                       variable_name=[f"{dike}_Dike Investment Costs" for dike in function.dikelist])
#     ]
#
#     dike_model.outcomes = outcomes
#
#     return dike_model

# Get the model for a specific problem formulation
problem_formulation_id = 6  # Change this to the desired problem formulation
dike_model = get_model_for_problem_formulation(problem_formulation_id)

if __name__ == '__main__':
    reference = Scenario('reference',
                         **{
                             'A.0_ID flood wave shape': 100,
                             'A.1_Bmax': 200, 'A.1_Brate': 1.5, 'A.1_pfail': 0.5,
                             'A.2_Bmax': 200, 'A.2_Brate': 1.5, 'A.2_pfail': 0.5,
                             'A.3_Bmax': 200, 'A.3_Brate': 1.5, 'A.3_pfail': 0.5,
                             'A.4_Bmax': 200, 'A.4_Brate': 1.5, 'A.4_pfail': 0.5,
                             'A.5_Bmax': 200, 'A.5_Brate': 1.5, 'A.5_pfail': 0.5,
                             'discount rate 0': 2.5, 'discount rate 1': 2.5, 'discount rate 2': 2.5
                         })

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"./archives_{timestamp}"

    os.makedirs(archive_dir, exist_ok=True)

    with MultiprocessingEvaluator(dike_model) as evaluator:
        # Define convergence metrics
        convergence_metrics = [
            ArchiveLogger(
                archive_dir,
                [l.name for l in dike_model.levers],
                [o.name for o in dike_model.outcomes],
                base_filename="single_run_2.tar.gz"
            ),
            EpsilonProgress(),
        ]

        # Run optimization
        result, convergence = evaluator.optimize(nfe=100000, reference=reference, epsilons=[0.1, 0.1, 0.1, 0.1],
                                                 convergence=convergence_metrics)

    # Save results
    result_df = pd.DataFrame(result)

    # Gather all relevant columns: uncertainties, levers, and outcomes
    uncertainty_columns = [u.name for u in dike_model.uncertainties]
    outcome_columns = [o.name for o in dike_model.outcomes]
    lever_columns = [l.name for l in dike_model.levers]

    result_df.to_csv('output/optimization_policies_singlerun.csv', index=False)
    outcomes_df = result_df.loc[:, [col for col in result_df.columns if col in [o.name for o in dike_model.outcomes]]]
    #outcomes without uncertainties
    outcomes_df.to_csv('output/optimization_outcomes_singlerun.csv', index=False)

    # Ensure all relevant columns are included in the DataFrame
    all_columns = uncertainty_columns + lever_columns + outcome_columns
    result_df_combined = result_df.loc[:, all_columns]

    # Save the combined DataFrame to CSV
    result_df_combined.to_csv('output/combined_optimization_outcomes_singlerun.csv', index=False)

    # Save convergence metrics
    convergence_df = pd.DataFrame(convergence)
    convergence_df.to_csv('output/convergence_metrics_singlerun.csv', index=False)


