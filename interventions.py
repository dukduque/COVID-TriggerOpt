'''
This module defines the knobs of an intervention and forms
the available intervantions considerin school closures,
cocooning, and different levels od social distance.
'''
from numpy import exp, round, array


class Intervension:
    def __init__(self, SC, CO, SD, epi, demographics):
        '''
            Attrs:
            school_closure (int): 0 schools are open, 1 schools are closed
            cocooning (float): level of cocooning [0,1)
            social_distance (float): level of social distance [0,1)
            epi (EpiParams): instance of the parameterization
            demographics (ndarray): Population demographics
        '''
        self.school_closure = SC
        self.cocooning = CO
        self.social_distance = SD
        self.cost = SC + CO + (round(exp(5 * SD), 3) - 1)
        demographics_normalized = demographics / demographics.sum()
        self.phi_weekday = epi.effective_phi(SC, CO, SD, demographics_normalized, weekday=True)
        self.phi_weekend = epi.effective_phi(SC, CO, SD, demographics_normalized, weekday=False)
    
    def phi(self, is_weekday):
        if is_weekday:
            return self.phi_weekday
        else:
            return self.phi_weekend
    
    @property
    def SC(self):
        return self.school_closure
    
    @property
    def CO(self):
        return self.cocooning
    
    @property
    def SD(self):
        return self.social_distance


def form_interventions(social_distance_levels, epi, demographics):
    interventions = []
    for SC in [0, 1]:
        for CO in [0, 1]:
            for SD in social_distance_levels:
                interventions.append(Intervension(SC, CO, SD, epi, demographics))
    return interventions
