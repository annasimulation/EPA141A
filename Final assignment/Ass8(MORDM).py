# Import necessary libraries
import pandas as pd
import numpy as np
from ema_workbench import (Model, RealParameter, IntegerParameter, CategoricalParameter,
                           ScalarOutcome, perform_experiments, Samplers, Policy, Scenario, HypervolumeMetric,
    GenerationalDistanceMetric,
    EpsilonIndicatorMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric,)
from ema_workbench.em_framework import (SequentialEvaluator, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import (EpsNSGAII, Convergence)
from ema_workbench.analysis import parcoords
from dike_model_function import DikeNetwork
import matplotlib.pyplot as plt
from ema_workbench.em_framework.optimization import (ArchiveLogger,
                                                     EpsilonProgress)
import os
from ema_workbench.em_framework.optimization import to_problem
import shutil


# Define the sum_over function
def sum_over(*args):
    numbers = []
    for entry in args:
        try:
            value = sum(entry)
        except TypeError:
            value = entry
        numbers.append(value)
    return sum(numbers)


# Define the function to get the model for the given problem formulation
def get_model_for_problem_formulation(problem_formulation_id):
    function = DikeNetwork()
    dike_model = Model("dikesnet", function=function)

    Real_uncert = {"Bmax": [30, 350], "pfail": [0, 1]}  # m and [.]
    cat_uncert_loc = {"Brate": (1.0, 1.5, 10)}
    cat_uncert = {f"discount rate {n}": (1.5, 2.5, 3.5, 4.5) for n in function.planning_steps}
    Int_uncert = {"A.0_ID flood wave shape": [0, 132]}

    uncertainties = []

    for uncert_name in cat_uncert.keys():
        categories = cat_uncert[uncert_name]
        uncertainties.append(CategoricalParameter(uncert_name, categories))

    for uncert_name in Int_uncert.keys():
        uncertainties.append(IntegerParameter(uncert_name, Int_uncert[uncert_name][0], Int_uncert[uncert_name][1]))

    dike_height_levers = []
    for dike in function.dikelist:
        for uncert_name in Real_uncert.keys():
            name = f"{dike}_{uncert_name}"
            lower, upper = Real_uncert[uncert_name]
            uncertainties.append(RealParameter(name, lower, upper))

        for uncert_name in cat_uncert_loc.keys():
            name = f"{dike}_{uncert_name}"
            categories = cat_uncert_loc[uncert_name]
            uncertainties.append(CategoricalParameter(name, categories))

        for n in function.planning_steps:
            name = f"{dike}_DikeIncrease {n}"
            dike_height_levers.append(IntegerParameter(name, 0, 10))

    dike_model.uncertainties = uncertainties

    dike_height_levers.append(IntegerParameter("EWS_DaysToThreat", 0, 4))  # days
    # Set levers: No RfR, dike heightening
    dike_model.levers = dike_height_levers

    # Define the outcomes
    outcomes = [
        ScalarOutcome('Total Costs', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
                                                                                                       f"{dike}_Expected Annual Damage"
                                                                                                       for dike in
                                                                                                       function.dikelist] +
                                                                                                   [
                                                                                                       f"{dike}_Dike Investment Costs"
                                                                                                       for dike in
                                                                                                       function.dikelist] +
                                                                                                   ["RfR Total Costs"]
                      ),
        ScalarOutcome('Expected Number of Deaths', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
            f"{dike}_Expected Number of Deaths" for dike in function.dikelist]
                      ),
        ScalarOutcome('Expected Annual Damage', kind=ScalarOutcome.MINIMIZE, function=sum_over, variable_name=[
            f"{dike}_Expected Annual Damage" for dike in function.dikelist]),
        ScalarOutcome('Dike Investment Costs', kind=ScalarOutcome.MINIMIZE, function=sum_over,
                      variable_name=[f"{dike}_Dike Investment Costs" for dike in function.dikelist])
    ]

    dike_model.outcomes = outcomes



    return dike_model

# Get the model for a specific problem formulation
problem_formulation_id = 6  # Change this to the desired problem formulation
dike_model = get_model_for_problem_formulation(problem_formulation_id)

if __name__ == '__main__':
    reference = Scenario('reference',
                         **{
        'A.0_ID flood wave shape':100,
        'A.1_Bmax':200, 'A.1_Brate':1.5, 'A.1_pfail':0.5,
        'A.2_Bmax':200, 'A.2_Brate':1.5, 'A.2_pfail':0.5,
        'A.3_Bmax':200, 'A.3_Brate':1.5, 'A.3_pfail':0.5,
        'A.4_Bmax':200, 'A.4_Brate':1.5, 'A.4_pfail':0.5,
        'A.5_Bmax':200, 'A.5_Brate':1.5, 'A.5_pfail':0.5,
        'discount rate 0':2.5, 'discount rate 1':2.5, 'discount rate 2':2.5
    })

    # Now proceed with defining the convergence metrics
    convergence_metrics = [ArchiveLogger(
        "./archives",
        [l.name for l in dike_model.levers],
        [o.name for o in dike_model.outcomes],
        base_filename="tutorial.tar.gz",
    ),
        EpsilonProgress(),
    ]

    # Configure logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    with MultiprocessingEvaluator(dike_model) as evaluator:
        results = evaluator.optimize(nfe=15, reference=reference, epsilons=[0.1, 0.1, 0.1, 0.1],
                                     convergence=convergence_metrics)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the decision variables (policies) and outcomes to separate CSV files
    results_policies = results_df.loc[:,
                       [col for col in results_df.columns if col in [l.name for l in dike_model.levers]]]
    results_outcomes = results_df.loc[:,
                       [col for col in results_df.columns if col in [o.name for o in dike_model.outcomes]]]

    results_policies.to_csv('optimization_policies.csv', index=False)
    results_outcomes.to_csv('optimization_outcomes.csv', index=False)

    # Plotting the results using parallel coordinates
    outcomes = results_df.loc[:,
               ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']]

    # Define limits for parallel coordinates plot
    limits = pd.DataFrame(
        [[outcomes[col].min() for col in outcomes.columns],
         [outcomes[col].max() for col in outcomes.columns]],
        columns=outcomes.columns
    )

    # Create parallel axes
    axes = parcoords.ParallelAxes(limits)

    # Plot the outcomes
    for i, color in enumerate(sns.color_palette(n_colors=len(outcomes))):
        axes.plot(outcomes, color=color, label='result {}'.format(i))

    # Invert the axis for outcomes where lower values are better
    axes.invert_axis('Total Costs')
    axes.invert_axis('Expected Number of Deaths')
    axes.invert_axis('Expected Annual Damage')
    axes.invert_axis('Dike Investment Costs')

    # Display the legend and plot
    axes.legend()
    plt.show()

    all_archives = []
    for i in range(5):
        archives = ArchiveLogger.load_archives(f"./archives/{i}.tar.gz")
        all_archives.append(archives)

    problem = to_problem(dike_model, searchover="levers")
    reference_set = epsilon_nondominated(results, epsilons=[0.125, 0.05, 0.01, 0.01], problem=problem)

    metrics_by_seed = []
    for entry in all_archives:
        metrics = calculate_metrics(entry, reference_set)
        metrics_by_seed.append(metrics)

    sns.set_style("white")
    fig, axes = plt.subplots(nrows=6, figsize=(8, 12), sharex=True)

    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    for metrics, convergence in zip(metrics_by_seed, convergences):
        ax1.plot(metrics.nfe, metrics.hypervolume)
        ax1.set_ylabel("hypervolume")

        ax2.plot(convergence.nfe, convergence.epsilon_progress)
        ax2.set_ylabel("$\epsilon$ progress")

        ax3.plot(metrics.nfe, metrics.generational_distance)
        ax3.set_ylabel("generational distance")

        ax4.plot(metrics.nfe, metrics.epsilon_indicator)
        ax4.set_ylabel("epsilon indicator")

        ax5.plot(metrics.nfe, metrics.inverted_gd)
        ax5.set_ylabel("inverted generational\ndistance")

        ax6.plot(metrics.nfe, metrics.spacing)
        ax6.set_ylabel("spacing")

    ax6.set_xlabel("nfe")

    sns.despine(fig)
    plt.show()