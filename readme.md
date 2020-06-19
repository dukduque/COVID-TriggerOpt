# Overview
This software support the results in the paper: 
https://www.medrxiv.org/content/10.1101/2020.04.29.20085134v1.article-metrics

The algorithm simulate a covid-19 epidemic for Austin, Tx, and de determines triggers
to enact social distancing orders to avoid exceeding hospital capacity. Chosen triggers
attempt to minimize the total number of days that a city is in lock-down. 

# Structure of the code

## threshold_policy.py
- Main module to launch the search
- Functions to excude the search and find optimized thresholds to enact lock-downs.
- Iterators for traing and testing
- Calendar generation (ad-hoc for Austin instance)

## SEIYHARD_sim.py:
- Simulator engine
- Parallelizarion functions
- Calander utils class (SimCalendar)

## epi_parameters.py
- Class EpiSetup to characterize the simulation and recompute contact matrices as needed.

## intervention.py
- Class intervention defining its properties and used in the simulator.
- Helper function to create multiple interventions

## utils.py
- Timing function
- Rounding functions

## intances/austin.py
- Module summarizing the input of the simulator.
- Creates an instance of EpiSetup


## Guidelines
- Create new branches to test new features
- Create a pull request to merge with master
