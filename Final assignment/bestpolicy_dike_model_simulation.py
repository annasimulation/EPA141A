from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
import time
from problem_formulation import get_model_for_problem_formulation
import pandas as pd


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(6)

    best_policy = {
        'A.1_DikeIncrease 0': 2.0,
        'A.1_DikeIncrease 1': 0.0,
        'A.1_DikeIncrease 2': 0.0,
        'A.2_DikeIncrease 0': 3.0,
        'A.2_DikeIncrease 1': 0.0,
        'A.2_DikeIncrease 2': 0.0,
        'A.3_DikeIncrease 0': 4.0,
        'A.3_DikeIncrease 1': 0.0,
        'A.3_DikeIncrease 2': 0.0,
        'A.4_DikeIncrease 0': 2.0,
        'A.4_DikeIncrease 1': 0.0,
        'A.4_DikeIncrease 2': 0.0,
        'A.5_DikeIncrease 0': 3.0,
        'A.5_DikeIncrease 1': 0.0,
        'A.5_DikeIncrease 2': 0.0,
        'EWS_DaysToThreat': 3.0,
    }

    # Create the policy object
    policy = Policy('Best Policy', **best_policy)
    # Build a user-defined scenario and policy:
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 2,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_steps})
    scen1 = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split("_")

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **scen1)


    # Call random scenarios or policies:
    # n_scenarios = 5
    # scenarios = sample_uncertainties(dike_model, 50)
    # n_policies = 10

    # single run
    #    start = time.time()
    #    dike_model.run_model(ref_scenario, policy0)
    #    end = time.time()
    #    print(end - start)
    #    results = dike_model.outcomes_output

# multiprocessing
    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.perform_experiments(scenarios=1000, policies=[policy])

    experiments, outcomes = results

    data = pd.DataFrame(experiments)
    for key, value in outcomes.items():
        data[key] = value

    data.to_csv('experiments_and_outcomes.csv', index=False)

    # Convert experiments and outcomes to dataframes
    experiments_df = pd.DataFrame(experiments)
    outcomes_df = pd.DataFrame(outcomes)

    # Display experiments
    experiments_df.to_csv('experiments_bestpolicy.csv', index=False)

    # Display outcomes
    outcomes_df.to_csv('outcomes_bestpolicy.csv', index=False)