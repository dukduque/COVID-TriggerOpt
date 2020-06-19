'''
    Instance for Austin metro area
'''

import numpy as np
import datetime as dt
from epi_params import EpiSetup
from SEIYAHRD_sim import SimCalendar
from scipy.optimize import root_scalar
city = 'austin'

# Demographics and hospitals data (Remy)
# RiskGroup,Metro,0-4,5-17,18-49,50-64,65+
# 0,12420,128527,327148,915894,249273,132505
# 1,12420,9350,37451,156209,108196,103763
N = np.array([[128527, 327148, 915894, 249273, 132505], [9350, 37451, 156209, 108196, 103763]],
             dtype=np.int).transpose()
population = N.sum()

A = 5  # number of age groups
L = 2  # number of risk groups

# Initial conditions
start_date = dt.datetime(year=2020, month=2, day=15)
I0 = np.zeros_like(N)
I0[2, 0] = 1  # Start with one case on Feb 15th

# Calendar and simulation length
end_date = dt.datetime(year=2021, month=9, day=30)
T = 1 + (end_date - start_date).days
cal = SimCalendar(start_date, T)
lockdown_start = dt.datetime(year=2020, month=3, day=24)
lockdown_end = dt.datetime(year=2020, month=5, day=1)
school_closure_end = dt.datetime(year=2020, month=8, day=18)
last_day_interventions = dt.datetime(year=2021, month=9, day=30)
cal.load_initial_lockdown(lockdown_start, lockdown_end)
cal.load_school_closure(lockdown_start, school_closure_end)

# 80% of 4,299 total beds so 3,239
hosp_beds = 3_239
icu = 675
ventilators = 675
root_sol = root_scalar(lambda x: x + 4 * np.sqrt(x) - hosp_beds, x0=hosp_beds, x1=hosp_beds * 0.5)
assert root_sol.converged, 'Staffing rule failed'
lambda_star = root_sol.root

# Epidemiological scenario
epi = EpiSetup(case_id=6)

# Additional changes for the instance
epi.beta = 0.0351073159954258

# Summary
ins_name = f'{city}_{T}'
summary = (epi, T, A, L, N, I0, hosp_beds, lambda_star, cal)
