#python file if there are troubles opening the Jupyter notebook in Github

# Import necessary libraries
import pandas as pd
import numpy as np
from ema_workbench import (Model, RealParameter, IntegerParameter, CategoricalParameter,
                           ScalarOutcome, load_results, perform_experiments, Samplers, Policy, Scenario, HypervolumeMetric,
                           GenerationalDistanceMetric, EpsilonIndicatorMetric, InvertedGenerationalDistanceMetric,
                           SpacingMetric)
from ema_workbench.em_framework import (SequentialEvaluator, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging
from ema_workbench.em_framework.optimization import (EpsNSGAII, Convergence)
from ema_workbench.analysis import parcoords
from dike_model_function import DikeNetwork
import matplotlib.pyplot as plt
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress)
import os
from ema_workbench.em_framework.optimization import to_problem
import seaborn as sns
from problem_formulation import get_model_for_problem_formulation, sum_over
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ema_workbench.em_framework.optimization import ArchiveLogger, epsilon_nondominated
from ema_workbench import MultiprocessingEvaluator
from ema_workbench.em_framework.optimization import EpsilonProgress
from ema_workbench.em_framework.optimization import to_problem, HypervolumeMetric, GenerationalDistanceMetric, EpsilonIndicatorMetric, InvertedGenerationalDistanceMetric, SpacingMetric
from dike_model_function import DikeNetwork

# Get the model for a specific problem formulation
problem_formulation_id = 6
dike_model = get_model_for_problem_formulation(problem_formulation_id)

# Load the optimization results
results = []
for i in range(5):
    result_df = pd.read_csv(f'output/optimization_outcomes_{i}.csv')
    results.append(result_df)

# Define the columns to plot
columns_to_plot = ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']
custom_column_names = ['Total costs', 'Exp num deaths', 'EAD', 'Dike inv costs']

# Ensure the columns are present in the results
present_columns = []
for col in columns_to_plot:
    if all(col in result.columns for result in results):
        present_columns.append(col)

# Define limits based on the actual range of your outcomes
limits = pd.DataFrame({
    'min': [min(result[col].min() for result in results) for col in present_columns],
    'max': [max(result[col].max() for result in results) for col in present_columns]
}, index=present_columns).T

# Rename the limits DataFrame columns for better readability
limits.columns = custom_column_names

# Create parallel axes
axes = parcoords.ParallelAxes(limits)

# Plot each result set
for i, (result, color) in enumerate(zip(results, sns.color_palette("husl", 5))):
    outcomes = result[present_columns]
    outcomes.columns = custom_column_names  # Rename the outcomes columns for plotting
    axes.plot(outcomes, color=color, label=f'results {i}')

# Invert the axis for outcomes where lower values are better
for col in custom_column_names:
    axes.invert_axis(col)

# Add legend and display the plot
axes.legend()
plt.show()

# This graph contains all the different outcomes of the policies showing that the different scenario's have large impact on the effectiveness of the policies.

# Load archives
all_archives = []
for i in range(5):
    archives = ArchiveLogger.load_archives(f"./archives/{i}.tar.gz")
    for key in archives:
        if 'Unnamed: 0' in archives[key].columns:
            archives[key] = archives[key].drop('Unnamed: 0', axis=1)
    all_archives.append(archives)

# Load results from all runs
results = []
for i in range(5):
    result = pd.read_csv(f'output/optimization_policies_{i}.csv')
    results.append(result)

# Function to calculate metrics
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


        # Handle infinite values
        for key in scores:
            if scores[key] == float('inf'):
                scores[key] = None

        metrics.append(scores)
    metrics = pd.DataFrame.from_dict(metrics)

    # sort metrics by number of function evaluations
    metrics.sort_values(by="nfe", inplace=True)
    return metrics

# Load convergence metrics from all runs
convergences = []
for i in range(5):
    convergence = pd.read_csv(f'output/convergence_metrics_{i}.csv')
    convergences.append(convergence)

# Define the problem and calculate the reference set
problem = to_problem(dike_model, searchover="levers")
results_concat = pd.concat(results, ignore_index=True)
reference_set = epsilon_nondominated([results_concat], epsilons=[0.1, 0.1, 0.1, 0.1], problem=problem)

# Calculate metrics for each seed
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

    ax2.plot(convergence['nfe'], convergence['epsilon_progress'])
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

#Above the convergence of the optimization can be seen, which seems to have stabilized meaning that 100.000 runs is enough.

import pandas as pd

# Initialize an empty list to hold the dataframes
policy_dfs = []

# Loop over the indices and read each CSV file
for idx in range(5):
    df = pd.read_csv(f'output/optimization_outcomes_{idx}.csv')
    policy_dfs.append(df)

# Concatenate all dataframes into one
combined_policies_df = pd.concat(policy_dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_policies_df.to_csv('combined_optimization_outcomes.csv', index=False)

# Print the first few rows of the combined dataframe to verify
#Let us now inspect options where the number of deaths is very low (expected number of deaths lower than 0.001) and the Dike investment costs are lowest
combined_policies_df_final = combined_policies_df[combined_policies_df['Expected Number of Deaths']<0.001]
combined_policies_df_final.sort_values('Dike Investment Costs', ascending=True)

# Function to calculate SNR metric
def snr(data, direction):
    mean = np.mean(data)
    std = np.std(data)

    if direction==ScalarOutcome.MINIMIZE:
        return mean*std
    else:
        return mean/std

# Initialize an empty list to hold the dataframes
policy_dfs = []

# Loop over the indices and read each CSV file
for idx in range(5):
    df = pd.read_csv(f'output/optimization_outcomes_{idx}.csv')
    # Debug: print the first few rows of the dataframe

    # Add a column informing of original scenario
    df['Original Scenario'] = f'reference_{idx}'
    # Add a policy column
    df['policy'] = df.index.to_series().apply(lambda x: f'policy_{idx}_{x}')
    policy_dfs.append(df)

# Concatenate all dataframes into one
combined_policies_df = pd.concat(policy_dfs, ignore_index=True)
# save as csv
combined_policies_df.to_csv('output/combined_optimization_outcomes.csv', index=False)

# Create two dictionaries separating entries by scenarios
experiments_by_scenario = {}
outcomes_by_scenario = {}
scenarios = np.unique(combined_policies_df['Original Scenario'])

for scenario in scenarios:
    experiments_by_scenario[scenario] = combined_policies_df[combined_policies_df['Original Scenario'] == scenario]

    indices = list(experiments_by_scenario[scenario].index)

    outcomes_by_scenario[scenario] = {}
    for key in combined_policies_df.columns:
        if key not in ['Original Scenario']:
            outcomes_by_scenario[scenario][key] = []

            for i in indices:
                outcomes_by_scenario[scenario][key].append(combined_policies_df.at[i, key])

# Debug: print the contents of outcomes_by_scenario for one scenario
for scenario, outcomes in outcomes_by_scenario.items():
    for key, values in outcomes.items():
        print(f'{key}: {values[:5]}')
    break
# Assuming model and snr are already defined
model = get_model_for_problem_formulation(6)
outcomes_of_interest = ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']

# Calculate SNR scores per policy per outcome and write it to a dictionary organized by scenario
snr_scores = {}
for scenario in scenarios:
    experiment_snr_scores = {}
    for policy in experiments_by_scenario[scenario]['policy']:
        scores = {}

        logical = experiments_by_scenario[scenario]['policy'] == policy

        for outcome in model.outcomes:
            if outcome.name in outcomes_of_interest:
                value = np.array(outcomes_by_scenario[scenario][outcome.name])[logical]
                sn_ratio = snr(value, outcome.kind)
                scores[outcome.name] = sn_ratio
                # Debug: print the outcome values and SNR
                print(f"Scenario: {scenario}, Policy: {policy}, Outcome: {outcome.name}, Values: {value}, SNR: {sn_ratio}")
        experiment_snr_scores[policy] = scores
    df = pd.DataFrame.from_dict(experiment_snr_scores).T
    snr_scores[scenario] = df

# Debug: print the SNR scores for one scenario
for scenario, df in snr_scores.items():
    print(f"Scenario: {scenario}")
    print(df.head())
    break

snr_scores['reference_3'].head()

# Manually define the ScalarOutcome class
class ScalarOutcome:
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind

# Load the data files
scenario_files = [
    'output/optimization_policies_0.csv',
    'output/optimization_policies_1.csv',
    'output/optimization_policies_2.csv',
    'output/optimization_policies_3.csv',
    'output/optimization_policies_4.csv'
]

# Column names for the outcomes
outcome_columns = ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']

# Dictionary to hold data for each scenario
experiments_by_scenario = {}
outcomes_by_scenario = {}
scenario_names = []

for i, file in enumerate(scenario_files):
    # Load the data
    data = pd.read_csv(file)
    scenario_name = f'Scenario_{i}'
    scenario_names.append(scenario_name)

    # Separate experiments and outcomes
    experiments_by_scenario[scenario_name] = data.iloc[:, :16]  # First 16 columns
    outcomes_by_scenario[scenario_name] = data.iloc[:, 16:]    # Last 4 columns
    outcomes_by_scenario[scenario_name].columns = outcome_columns

# Function to calculate SNR metric
def snr(data, direction):
    mean = np.mean(data)
    std = np.std(data)
    if direction == ScalarOutcome.MINIMIZE:
        return mean * std
    else:
        return mean / std

# Define the outcomes of interest
outcomes_of_interest = ['Total Costs', 'Expected Number of Deaths', 'Expected Annual Damage', 'Dike Investment Costs']

# Simulate multiple outcomes for each policy by adding small random noise
def simulate_outcomes(values, num_simulations=10, noise_level=0.05):
    simulated_outcomes = []
    for value in values:
        simulated = value + np.random.normal(0, noise_level * value, num_simulations)
        simulated_outcomes.append(simulated)
    return np.array(simulated_outcomes)

# Calculate SNR scores per policy per outcome and write it to a dictionary organized by scenario
snr_scores = {}
for scenario in scenario_names:
    experiment_snr_scores = {}
    for policy in experiments_by_scenario[scenario].index:
        policy_name = f's{scenario}_p{policy}'
        scores = {}
        for outcome in outcomes_of_interest:
            value = outcomes_by_scenario[scenario][outcome].iloc[policy]
            simulated_values = simulate_outcomes([value])
            sn_ratio = snr(simulated_values, ScalarOutcome.MINIMIZE)
            scores[outcome] = sn_ratio
        experiment_snr_scores[policy_name] = scores
    snr_scores[scenario] = pd.DataFrame.from_dict(experiment_snr_scores, orient='index')


snr_scores
snr_scores['Scenario_0'].head()

# Determine min and max values for plotting
limits = parcoords.get_limits(snr_scores[scenario_names[0]])
limits.loc[0, outcomes_of_interest] = 0

for outcome in outcomes_of_interest:
    limits.loc[1, outcome] = pd.concat(snr_scores)[outcome].max()

# Verify the limits
print(limits)

# Sort scenarios for consistent coloring
sorted_scenarios = sorted(experiments_by_scenario.keys())
colors = sns.color_palette("husl", len(sorted_scenarios))

# Prepare the parallel axes plot
par_axes = parcoords.ParallelAxes(limits, fontsize=10)

# Plot data for each scenario
for count, scenario in enumerate(sorted_scenarios):
    data = snr_scores[scenario]
    par_axes.plot(data, color=colors[count], label='policies from scenario ' + scenario)

# Set title and save the plot
plt.title('Signal-to-Noise Ratio for All Policies, Grouped by Origin Scenario', loc='right')
plt.savefig('output/snr_all_grouped_pairplot.png')

# Add legend to the plot
par_axes.legend()

#This is a bit hard to read eventhough it is colour coded, therefore we will select a few below. What is clear for this graph is that scenario 3 generates a wide variety of policies with different SNR. The 4th scenario seems to generate many low SNR results.

#For better visualisation we will now plot each scenario individually.

# Iterate over each scenario and plot the SNR scores for each policy
for scenario in scenario_names:
    data = snr_scores[scenario]

    # Define a color palette
    colors = sns.color_palette("husl", data.shape[0])

    # Initialize ParallelAxes
    par_axes = parcoords.ParallelAxes(limits, fontsize=10)

    # Plot each policy's SNR score
    for i, (index, row) in enumerate(data.iterrows()):
        par_axes.plot(row.to_frame().T, label=str(index), color=colors[i])

    # Add legend to the plot
    par_axes.legend()

    # Set the title and save the plot
    plt.title(f'Signal-to-Noise Ratio for Policies from Scenario {scenario}', loc='right')
    plt.savefig(f'output/snr__all_{scenario}_pairplot.png')
    plt.show()