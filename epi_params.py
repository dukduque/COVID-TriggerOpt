'''
Epidemiology parameters from TACC simulation model
Author of the epi model: Zhanwei Du
'''
import numpy as np
from itertools import product


class EpiSetup:
    '''
        A setup for the epidemiological parameters.
        Scenarios 1--5 come from CDC. Scenarios 6 and 7
        correspond to best guess parameters for UT group.
    '''
    def __init__(self, case_id):
        '''
            Initialize an instance of epidemiological parameters. If the
            parameter is random, is not initialize and is queried as a
            property

            Args:
             case_id (int): case being run.
        '''
        self.case = case_id
        self.rnd_stream = None
        assert 0 <= case_id <= 6, 'Case should be between 0 and 6'
        # Transmission rate
        self.beta = [0.031, 0.023, 0.067, 0.044, 0.044, 0.01622242, 0.02599555][case_id]
        
        # Symptomatic fraction
        self.tau = [0.80, 0.50, 0.80, 0.50, 0.50, 0.821, 0.821][case_id]
        
        # Proportion of pre-symptomatic (%)
        self.pp = [0.20, 0.50, 0.20, 0.50, 0.50, 0.126, 0.126][case_id]
        
        # Exposed rate. Cases 1-5: 1/T(2,5,8), Cases 6-7: 1/T(5.6,7,8.2)
        # Using mean values for intervention models.
        self._sigma_E = None  # Computed as property
        
        # Infectioness scenarios, by age only for omega_E
        self._omega_E = [[0.04210526, 0.26923077, 0.04210526, 0.26923077, 0.26923077][case_id]
                         ] * 5 if case_id <= 4 else None  # Computed as property
        self.omega_IA = [0.50, 1, 0.500, 1, 1, 0.4653, 0.4653][case_id]
        self.omega_IY = 1
        
        # Recovery rates for each age group T(21.2, 22.6, 24.4)
        self._gamma_IY = None  # Computed as property
        self._gamma_IA = self._gamma_IY
        self.gamma_IH = [0.125, 0.125, 0.111, 0.1, 0.1] if case_id <= 4 else [1 / 14] * 5
        
        # symptomatic case hospitalization ratio (%) by age group
        # Best guess for scenarios 6 and 7
        YHR_Guess = np.array([
            [0.0003, 0.0002, 0.0132, 0.0286, 0.0339],
            [0.0028, 0.0022, 0.1320, 0.2860, 0.3390],
        ]).transpose()
        # 5 scenarios (rows) for each of the 5 age groups (cols)
        YHR = 0.01 * np.array([
            [0.7, 0.25, 0.5, 1, 9],
            [0.7, 0.25, 0.5, 1, 9],
            [5., 2., 5., 7., 60.],
            [5., 2., 5., 7., 60.],
            [1.25, 0.5, 1.25, 1.75, 16.],
        ])[case_id] if case_id <= 4 else YHR_Guess
        self.YHR = YHR
        
        # Rate from symptom onset to hospitalized
        self.Eta = [0.32154341, 0.3003003, 0.3003003, 0.28011204, 0.29673591] if case_id <= 4 else [1 / 5.9] * 5
        
        # Rate from hospitalized to death
        self.mu = [0.1803046, 0.18775217, 0.07151878, 0.07276781, 0.07172397] if case_id <= 4 else [1 / 14] * 5
        
        # Rate of symptomatic to hospital
        self._pi = None  # Computed as property
        # Death rate
        self.death = np.array(
            [[0.00595799, 0.01070961, 0.07558743, 0.10025368, 0.14841381],
             [0.00595799, 0.01070961, 0.07558743, 0.10025368, 0.14841381],
             [0.00555981, 0.01003689, 0.05483651, 0.09560398, 0.1555082],
             [0.00555981, 0.01003689, 0.05483651, 0.09560398, 0.1555082],
             [0.00555981, 0.01003689, 0.05483651, 0.07688352, 0.1461907]][case_id] if case_id <= 4 else
            [[0.0390, 0.1208, 0.0304, 0.1049, 0.2269], [0.0390, 0.1208, 0.0304, 0.1049, 0.2269]]).transpose()
        
        # Contact matrices
        self.phi_all = np.array([
            [2.1600, 2.1600, 4.1200, 0.8090, 0.2810],
            [0.5970, 8.1500, 5.4100, 0.7370, 0.2260],
            [0.3820, 2.4300, 10.2000, 1.7000, 0.2100],
            [0.3520, 1.8900, 6.7100, 3.0600, 0.5000],
            [0.1900, 0.8930, 2.3900, 1.2000, 1.2300],
        ])
        
        self.phi_school = np.array([
            [0.9950, 0.4920, 0.3830, 0.0582, 0.0015],
            [0.1680, 3.7200, 0.9260, 0.0879, 0.0025],
            [0.0428, 0.6750, 0.8060, 0.0456, 0.0026],
            [0.0842, 0.7850, 0.4580, 0.0784, 0.0059],
            [0.0063, 0.0425, 0.0512, 0.0353, 0.0254],
        ])
        
        self.phi_work = np.array([
            [0, 0, 0, 0, 0.0000121],
            [0, 0.0787, 0.4340000, 0.0499, 0.0003990],
            [0, 0.181, 4.490, 0.842, 0.00772],
            [0, 0.131, 2.780, 0.889, 0.00731],
            [0.00000261, 0.0034900, 0.0706000, 0.0247, 0.0002830],
        ])
    
    def update_rnd_stream(self, rnd_stream):
        '''
            Generates random parametes from a given random stream.
            Coupled paramters are updated as well.
            Args:
                rnd_stream (RandomState): a RandomState instance from numpy.
        '''
        np.random.RandomState
        self.rnd_stream = rnd_stream
        if rnd_stream is None:
            # Assume worst paramters when doing deterministic simulation
            self._sigma_E = 1 / 7
            self._gamma_IY = np.array([1 / 22.6] * 5)
            self._gamma_IA = self._gamma_IY
        else:
            self._sigma_E = 1 / rnd_stream.triangular(5.6, 7, 8.2)
            self._gamma_IY = np.array([1 / rnd_stream.triangular(21.2, 22.6, 24.4)] * 5)
            self._gamma_IA = self._gamma_IY
        YHR = self.YHR
        self._omega_E = np.array([((YHR[a] / self.Eta[a]) +
                                   ((1 - YHR[a]) / self._gamma_IY[a])) * self.omega_IY * self._sigma_E * self.pp /
                                  (1 - self.pp) for a in range(len(YHR))])
        self._pi = np.array([
            YHR[a] * self._gamma_IY[a] / (self.Eta[a] + (self._gamma_IY[a] - self.Eta[a]) * YHR[a])
            for a in range(len(YHR))
        ])
    
    @property
    def sigma_E(self):
        return self._sigma_E
    
    @property
    def gamma_IY(self):
        return self._gamma_IY
    
    @property
    def gamma_IA(self):
        return self._gamma_IA
    
    @property
    def omega_E(self):
        return self._omega_E
    
    @property
    def pi(self):
        return self._pi
    
    def effective_phi(self, school, cocooning, social_distance, demographics, weekday):
        '''
            school (int): yes (1) / no (0) schools are closed
            cocoonign (float): percentage of transmition reduction [0,1]
            social_distance (int): percentage of social distance (0,1)
            demographics (ndarray): demographics by age and risk group
        '''
        
        A = len(demographics)  # number of age groups
        L = len(demographics[0])  # number of risk groups
        d = demographics  # A x L demographic data
        phi_all_extended = np.zeros((A, A, L, L))
        phi_school_extended = np.zeros((A, A, L, L))
        phi_work_extended = np.zeros((A, A, L, L))
        for a, b in product(range(A), range(A)):
            phi_ab_split = np.array([
                [d[b, 0], d[b, 1]],
                [d[b, 0], d[b, 1]],
            ])
            phi_ab_split = phi_ab_split / phi_ab_split.sum(1)
            phi_all_extended[a, b] = self.phi_all[a, b] * phi_ab_split
            phi_school_extended[a, b] = self.phi_school[a, b] * phi_ab_split
            phi_work_extended[a, b] = self.phi_work[a, b] * phi_ab_split
        
        # Apply school closure and social distance
        if weekday:
            phi_age_risk = (1 - social_distance) * (phi_all_extended - school * phi_school_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = phi_all_extended - school * phi_school_extended
                phi_age_risk[:, :, 1, :] = (1 - cocooning) * phi_age_risk_copy[:, :, 1, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
        else:  # is a weekend
            phi_age_risk = (1 - social_distance) * (phi_all_extended - phi_school_extended - phi_work_extended)
            if cocooning > 0:
                # Assumes 95% reduction on last age group and high risk
                # High risk cocooning
                phi_age_risk_copy = (phi_all_extended - phi_school_extended - phi_work_extended)
                phi_age_risk[:, :, 1, :] = (1 - cocooning) * phi_age_risk_copy[:, :, 1, :]
                # last age group cocooning
                phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
            assert (phi_age_risk >= 0).all()
            return phi_age_risk
