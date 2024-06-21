import matplotlib.pyplot as plt

from ema_workbench import (
    HypervolumeMetric,
    GenerationalDistanceMetric,
    EpsilonIndicatorMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric,
)
from ema_workbench.em_framework.optimization import to_problem
import pandas as pd
import numpy as np
from ema_workbench import (Model, RealParameter, IntegerParameter, CategoricalParameter,
                           ScalarOutcome, perform_experiments, Samplers, Policy, Scenario)
from ema_workbench.em_framework import (SequentialEvaluator, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import (EpsNSGAII, Convergence)
from ema_workbench.em_framework.optimization import (ArchiveLogger,Problem, EpsilonProgress)
from ema_workbench.analysis import parcoords
from dike_model_function import DikeNetwork
from problem_formulation import get_model_for_problem_formulation, sum_over

# Get the model for a specific problem formulation
problem_formulation_id = 6  # Change this to the desired problem formulation
dike_model = get_model_for_problem_formulation(problem_formulation_id)

# Load convergence metrics
convergence = pd.read_csv('convergence_metrics_singlerun.csv')

# Load archives
archives = ArchiveLogger.load_archives(f"./archives/single_run.tar.gz")

# Prepare the data
for key in archives:
    archives[key] = archives[key].drop('Unnamed: 0', axis=1)

reference_set = archives[max(archives.keys())]  # this is the final archive

# Calculate metrics
def calculate_metrics(archives, reference_set):
    problem = to_problem(dike_model, searchover="levers")

    hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    sm = SpacingMetric(problem)

    metrics = []
    for nfe, archive in archives.items():
        scores = {
            "generational_distance": gd.calculate(archive),
            "hypervolume": hv.calculate(archive),
            "epsilon_indicator": ei.calculate(archive),
            "inverted_gd": ig.calculate(archive),
            "spacing": sm.calculate(archive),
            "nfe": int(nfe),
        }
        metrics.append(scores)
    metrics = pd.DataFrame.from_dict(metrics)

    # sort metrics by number of function evaluations
    metrics.sort_values(by="nfe", inplace=True)
    return metrics

metrics = calculate_metrics(archives, reference_set)

# Plot metrics
def plot_metrics(metrics, convergence):
    sns.set_style("white")
    fig, axes = plt.subplots(nrows=6, figsize=(8, 12), sharex=True)

    ax1, ax2, ax3, ax4, ax5, ax6 = axes

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

# Plot the metrics
plot_metrics(metrics, convergence)

plt.show()
plt.savefig(f'img/{convergence}_for_{metrics}.png')


import pandas as pd
import seaborn as sns
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt

# Load the optimization results
result_df = pd.read_csv('optimization_outcomes_singlerun.csv')

# Define the columns to plot
columns_to_plot = ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']
custom_column_names = ['Total costs', 'Exp num deaths', 'EAD', 'Dike inv costs']

# Ensure the columns are present in the results
present_columns = [col for col in columns_to_plot if col in result_df.columns]

# Rename the outcomes columns for better readability
outcomes = result_df[present_columns]
outcomes.columns = custom_column_names

# Define limits for parallel coordinates plot
limits = parcoords.get_limits(outcomes)

# Create parallel axes
axes = parcoords.ParallelAxes(limits)

# Plot the outcomes
axes.plot(outcomes)

# Invert the axis for outcomes where lower values are better
for col in custom_column_names:
    axes.invert_axis(col)

# Display the plot
plt.show()
plt.savefig('img/optimization_MORDM_single.png')

result_df.head(10)

#Let us now inspect options where the number of deaths is very low (expected number of deaths lower than 0.001) and the Dike investment costs are lowest
result_final = result_df[result_df['Expected Number of Deaths']<0.001]
result_final.sort_values('Dike Investment Costs', ascending=True)
result_final.to_csv('results_final_MORDM_single.csv')

import pandas as pd

# Load the DataFrame
policy_df = pd.read_csv('optimization_policies_singlerun.csv')

print('The policy outcome is:')
# Print the column names and values of the first row, except the last four columns
first_row = policy_df.iloc[0]
for column, value in first_row[:-4].items():
    print(f"{column}: {value}")

#Resulting from this we see that the following policy is deemed as most optimised for the Transport company.

policy_df = pd.read_csv('optimization_outcomes_singlerun.csv', index_col=None)
print(f"Shape of data: {policy_df.shape[0]} rows, {policy_df.shape[1]} columns.")
policy_df.head(15)