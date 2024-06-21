# EPA141A Model-based Decision-making
This `README.md` is for the submission of the final assignment for the course *EPA141A Model-Based Decision Making*. 
The file will contain the contents of this folder and will give detailed descriptions on how to run the analyses in the files provided.
Further, it will elaborate on the steps performed for the analyses, as also described in the methodology of the report.

*Group 13:*

| Student            | Student Number |
|--------------------|----------------|
| Klessens, Joost    | 4913299        |
| Vos, Max           | 5083168        |
| ter Avest, Noë     | 4955889        |
| Paardekooper, Anna | 4909445        |

## Contents
- [Requirements](#requirements)
- [File Structure](#file-structure)
  - [Directories](#directories)
- [Structure of Analyses](#structure-of-analyses)
  - [Step 1: Exploratory Modelling and Analysis](#step-1-exploratory-modelling-and-analysis-)
  - [Step 2: Global Sensitivity Analysis](#step-2-global-sensitivity-analysis)
    - [Sobol Analysis](#sobol-analysis)
    - [Feature scoring - Extra Trees](#feature-scoring---extra-trees)
  - [Step 3: Many-Objective Robust Decision Making Single-Scenario](#step-3-many-objective-robust-decision-making-single-scenario)
  - [Step 4: Scenario Discovery](#step-4-scenario-discovery)
    - [Simulations for preferred policy](#simulations-for-preferred-policy)
    - [PRIM Analysis](#prim-analysis)
  - [Step 5: Many-Objective Robust Decision Making Multi-Scenario](#step-5-many-objective-robust-decision-making-multi-scenario)

## Requirements
The final assignment uses certain packages, of which the ema_workbench. All packages and version can be found in the `requirements.txt` file.
The packages can be installed inputting the following code in the terminal.
```
pip install -r requirements.txt
```
For running all the files, we used a virtual environment, using Python 3.11

## File Structure
Below, a tree can be found with all the directories found in the project. The structure consists of certain (output) folders and the main Python files used for analyses.
```
# Directories
├── archives/
├── data/
├── delftblue_outcomes_tests/
├── img/
├── output/
# Model files
├── __init__.py
├── dike_model_function.py
├── funs_dikes.py
├── funs_economy.py
├── funs_generate_network.py
├── funs_hydrostat.py
# Problem formulation
├── problem_formulation.py
# Analyses files 
# Per step performed in the analysis
├── dike_model_simulation.py
├── Sobol_Analysis.ipynb
├── MORDM_Singlerun.py
├── MORDM_Singlerun_results.ipynb
├── Bestpolicy_dike_model_simulation.py
├── Policy_Vulnerability_PRIM.ipynb
├── MORDM multi-scenario_results.ipynb
# Other
├── .gitattributes
├── .gitignore
├── __init__.py
├── LICENSE 
├── requirements.txt
└── README.md
```

### Directories
* [archives/](archives) In the archives folder, firstly, the tar files can be found from the MORDM analyses. Secondly, analyses have been stored for experimenting that where not further used. Also, certain outputs have been stored for potential further use.  
* [data/](data) The data folder contains the data provided by the course. No adjustments have been made in this folder, only files have been read.
* [delftblue_outcomest_tests/](img) The delftblue folders contains the files and outputs that have been used in the testing with the Delftblue supercomputer. Unfortunately, the running with the Delftblue computer did not supply us with the wished for outcomes of analysis. The files and scripts had been saved for potential further use. Unfortunately, there was too little time left to do this. 
* [img/](img) The images folder contains all the plotted figures and graphs from the analyses.
* [output/](output) The output folder contains all the data generated from the analysis files.

## Structure of Analyses
Below, the steps can be found in which order the analyses have been performed. A detailed description of all methods and uses can be found in the main report. This section will function as description how the running and the output of the files works. 

### Step 1: Exploratory Modelling and Analysis 
Initial exploratory modelling and analysis has been done using the provided dike files. First, an initial exploration has been done by investigating all files.
The problem formulation has been depicted in the [Problem Formulation](problem_formulation.py) file.
The `get_model_for_problem_formulation` function is called from multiple Python files to call the defined problem formulation.
The [Simulation file](dike_model_simulation.py) is used to create simulations for experiments, using different levers and uncertainties. 
The simulations can be step up and are saved in the `output` folder. 

### Step 2: Global Sensitivity Analysis
The global sensitivity analysis consists of two parts complementing each other. First, a Sobol analysis is performed, follow by a feature scoring analysis.
The file [Sobol Analysis](Sobol_Analysis.ipynb) is used for both parts of the global sensitivity analysis. 
#### Sobol Analysis
In the Sobol analysis, different policies of dike increases are tested against the outcomes and the Sobol analysis shows the influence of the uncertainties on the outcome `Expected Annual Damage`.
The output of the Sobol Analysis is saved as figures in the [img](img) folder and as data in the [output](output) folder. The data on the S1 and ST outputs are of importance to identify which uncertainties create most variance for the outcomes.  

#### Feature scoring - Extra Trees
To further validate the sensitivity analysis, feature scoring is performed to have valid answers to which uncertainties create most variance for our outcome of interest `Expected Annual Damage`.
The method of `Extra Trees` is chosen using a regression mode. Again, the figures are saved in the [img](img) folder and the data in the [output](output) folder.

### Step 3: Many-Objective Robust Decision Making Single-Scenario
Single scenario MORDM is performed in the [MORDM Multi Scenario](MORDM_Singlerun.py) file to find the most optimal policy. 
The optimal policy can be found in [optimization_policies_singlerun.csv](output/optimization_policies_singlerun.csv). The outcomes of the MORDM analysis are to be found in [optimization_outcomes_singlerun.csv](output/optimization_outcomes_singlerun.csv). 
Consequently, the convergence metrics are created and stored to [convergence_metrics_singlerun.csv](output/convergence_metrics_singlerun.csv).
The file [MORDM_Singlerun_results.ipynb](MORDM_Singlerun_results.ipynb) creates the results for the single scenario MORDM that are again saved in the [output](output) and [image](img) folders.  

### Step 4: Scenario Discovery
After the single scenario MORDM, the scenario discovery is performed, using the method of PRIM analysis. 
#### Simulations for preferred policy
A replication is made of the [dike_model_simulation.py](dike_model_simulation.py) file to create simulations for the optimised policy found in the MORDM analysis. These experiments are created using the [Bestpolicy_dike_model_simulation.py](Bestpolicy_dike_model_simulation.py) file. The files are saved as [experiments_bestpolicy.csv](output/experiments_bestpolicy.csv) and [outcomes_bestpolicy.csv](output/outcomes_bestpolicy.csv).

#### PRIM Analysis
Using these experiments from the preferred policy, a scenario discovery is done using PRIM. The goal is to find the 5 worst case scenarios for communication to the client and to use in the multi-scenario MORDM analysis.
The [Policy_Vulnerability_PRIM.ipynb](Policy_Vulnerability_PRIM.ipynb) uses the input from the simulations described before this. The graphs outputted describe the boxes where the worst case scenarios can be found and these are saved to the [images](img) folder. 
The boxes are further peeled to find the eventual [worst case scenarios](output/worst_5_case_scenarios.csv).

### Step 5: Many-Objective Robust Decision Making Multi-Scenario
Following, the 5 worst case scenarios are used for the MORDM for multi scenarios. This MORDM analysis runs the 5 worst case scenarios and further optimalises the preferred policies. Output can be found ... 
