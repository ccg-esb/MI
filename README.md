# Modelling Evolutionary Dynamics of Mobile Integrons

This repository contains a series of Jupyter notebooks that model the regulatory and population dynamics of mobile integrons. The notebooks are organized sequentially, beginning with the regulatory model and advancing to simulations of evolutionary dynamics. Each notebook focuses on a specific aspect of mobile integron biology, integrating theoretical frameworks with computational simulations to explore cassette shuffling, genotype accessibility, and population-level adaptations.

## Notebooks Overview

1. [**MI_1_RegulatoryModel.ipynb**](MI_1_RegulatoryModel.ipynb)  
   This notebook introduces the regulatory model of mobile integrons, focusing on the mechanisms controlling cassette shuffling and expression dynamics. It provides a foundation for understanding how integron polarity contribute to bacterial adaptability.

2. [**MI_2_RegulatoryModel_Data.ipynb**](MI_2_RegulatoryModel_Data.ipynb)  
   This notebook uses the regulatory model to generate synthetic data representing theoretical cassette expression and shuffling dynamics. Gene expression data from two-cassette integrons are extrapolated to simulate three-cassette systems for all possible cassette permutations.

3. [**MI_3_PopulationModel_Parameters.ipynb**](MI_3_PopulationModel_Parameters.ipynb)  
   This notebook defines key parameters for the population dynamics model, such as growth rates, resource utilization, antibiotic effects, and transition matrices. These parameters, derived from experimental data, serve as inputs for the subsequent simulations.

4. [**MI_4_PopulationModel_Competition.ipynb**](MI_4_PopulationModel_Competition.ipynb)  
   This notebook simulates competition between bacterial strains with varying integron configurations. Pairwise competition experiments are used to derive relative fitness networks, providing insights into how selective pressures, such as antibiotics, shape strain dynamics and fitness relationships.

5. [**MI_5_PopulationModel_Paths.ipynb**](MI_5_PopulationModel_Paths.ipynb)  
   This notebook analyzes evolutionary paths through genotype space, quantifying accessibility and fitness landscapes. It uses graph-based approaches to evaluate how integron shuffling affects evolutionary trajectories.

6. [**MI_6_PopulationModel_Evo.ipynb**](MI_6_PopulationModel_Evo.ipynb)  
   This notebook simulates long-term evolutionary dynamics in serial dilution experiments under constant environmental conditions. It evaluates how antibiotic treatments influence population dynamics and genetic diversity associated with mobile integrons.

7. [**MI_7_PopulationModel_FluctuatingEvo.ipynb**](MI_7_PopulationModel_FluctuatingEvo.ipynb)  
   This notebook investigates evolutionary dynamics under fluctuating selective pressures, such as antibiotic ramps and periodic antibiotic treatments. It explores how environmental variability influences genotype stability and evolutionary trajectories.

## Code

The implementation of core simulation functions is provided in [**code/MI_GillespieModel.py**](code/MI_GillespieModel.py). This script contains the Gillespie algorithm and associated functions used across the notebooks to model the dynamics of bacterial populations.

## Authors

[@Evolutionary Systems Biology, CCG-UNAM](http://www.penamiller.com/)  
[@Molecular Biology of Adaptation, Universidad Complutense](https://ucm.es/mbalab)


