# ----------------------------------------------------------------------------
# Contributors: Thiago O. Bezerra
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# File description:
#
# Functions implementing the equations for of the simplified astrocyte model, 
# functions to calculate the pre-synaptic spike times, and implement the 
# 4th-order Runge-Kutta method to solve the system of differential equations.
# ----------------------------------------------------------------------------

from numba import njit
import numpy as np

@njit
def calc_alpha_CaER(r_ER):    
    return (-1.03006013e-04/r_ER + 1.00099667e+00)/r_ER

@njit
def calc_beta_CaER(r_ER):
    return quartic(r_ER, -2.61407715e+02, 1.10942956e+02, -1.59939929e+01, 8.92587873e-01, 1.39080128e-02)*1369.0

@njit
def quartic(x, a, b, c, d, e): return a*x**4 + b*x**3 + c*x**2 + d*x + e

@njit
def calculate_caer(r_ER, c):
        
    a = quartic(r_ER, -2.61407715e+02, 1.10942956e+02, -1.59939929e+01, 8.92587873e-01, 1.39080128e-02)*1369.0
    b = -1.03006013e-04/r_ER + 1.00099667e+00
        
    return a - b/r_ER*c

@njit
def diffusion(M, D_coeff, mol):        
    return D_coeff*(M.dot(mol) - M.sum(axis=1)*mol)

@njit
def f_prod_PLCb_glu(v_beta, g, alpha, K_R, K_p, c, K_pi):
    return v_beta * g ** alpha / (g ** alpha + (K_R + K_p * c / (c + K_pi)) ** alpha)

@njit
def f_prod_PLCb_DA(v_DA, d, beta, K_DA, K_p, c, K_pi):
     return v_DA * d ** beta / (d ** beta + (K_DA + K_p * c / (c + K_pi))**beta)

@njit
def f_prod_PLCd(v_delta, i, kappa_delta, c, K_PLCdelta): 
    return v_delta / (1 + i/kappa_delta) * c ** 2 / (c ** 2 + K_PLCdelta ** 2)

@njit
def f_degr_IP3_3K(v_3K, c, K_D, i, K_3):
    return v_3K * c ** 4 / (c ** 4 + K_D ** 4) * i / (i + K_3)

@njit
def f_degr_IP_5P(r_5P, i):
    return r_5P * i

@njit
def f_I_IP3R(F, A, Vol, r_ER, r_C, i, d_1, alpha, c, d_5, h):
    
    h = h**3
    Ca_ER = alpha*calculate_caer(r_ER, c)
        
    return F * Vol / A * r_C * h * (i/(i+d_1))**3 * (c/(c+d_5))**3 * (Ca_ER-c)

@njit
def f_I_CERleak(F, A, Vol, r_ER, r_L, alpha, c):
    
    Ca_ER = alpha*calculate_caer(r_ER, c)
    
    return F * Vol / A * r_L * (Ca_ER - c)

@njit
def f_I_SERCA(F, A, Vol, v_ER, c, K_ER):
    return F * Vol / A * v_ER * c ** 2 / (c ** 2 + K_ER ** 2)

@njit
def f_I_NCX(alpha_NCX, beta_NCX, c):
    return beta_NCX - alpha_NCX * c

### ODEs ###
@njit
def dCa_idt(M, p, A, Vol, r_ER, Ca_ERalpha, c, i): 
                
    J_diff_Cai  = diffusion(M, p['D_Ca'], c)
    
    I_NCX     = f_I_NCX(p['alpha_NCX'], p['beta_NCX'], c)
    I_IP3R    = f_I_IP3R (p['F'], A, Vol, r_ER, p['r_C'], i, p['d_1'], Ca_ERalpha, c, p['d_5'], p['h'])
    I_SERCA   = f_I_SERCA(p['F'], A, Vol, p['v_ER'], c, p['K_ER'])
    I_CERleak = f_I_CERleak(p['F'], A, Vol, r_ER, p['r_L'], Ca_ERalpha, c)

    return A/(Vol*p['F'])*I_NCX + A*np.sqrt(r_ER)/(Vol*p['F'])*(I_IP3R - I_SERCA + I_CERleak) + J_diff_Cai

@njit
def dIP3dt(M, p, c, i, g, d):
    
    J_diff_IP3  = diffusion(M, p['D_IP3'], i)
    
    prod_PLCb_glu = f_prod_PLCb_glu(p['v_beta'], g, p['alpha'], p['K_R'], p['K_p'], c, p['K_pi'])
    prod_PLCb_DA  = f_prod_PLCb_DA(p['v_DA'], d, p['beta'], p['K_DA'], p['K_p'], c, p['K_pi'])
    prod_PLCd     = f_prod_PLCd(p['v_delta'], i, p['kappa_delta'], c, p['K_PLCdelta'])
    degr_IP3_3K   = f_degr_IP3_3K(p['v_3K'], c, p['K_D'], i, p['K_3'])
    degr_IP_5P    = f_degr_IP_5P(p['r_5P'], i)
    
    return prod_PLCb_glu + prod_PLCd + prod_PLCb_DA - degr_IP3_3K - degr_IP_5P + J_diff_IP3

@njit
def dGlu_odt(M, p, g):
    
    J_diff_Gluo = diffusion(M, p['D_glu'], g)
    
    return -p['G_glu'] * g + J_diff_Gluo

@njit
def dDA_odt(M, p, d):
    
    J_diff_DAo = diffusion(M, p['D_DA'], d)
    
    return - p['G_DA'] * d + J_diff_DAo

@njit
def model_eqs(M, p, A, Vol, r_ER, Ca_ERalpha, c, i, g, d):
    return np.vstack((dCa_idt(M, p, A, Vol, r_ER, Ca_ERalpha, c, i),
                      dIP3dt(M, p, c, i, g, d),
                      dGlu_odt(M, p, g),
                      dDA_odt(M, p, d)))

# Stationary Values

def null_IP3(i, p):

    prod_PLCb_glu = f_prod_PLCb_glu(p['v_beta'], p['g_rest'], p['alpha'], p['K_R'], p['K_p'], p['c_rest'], p['K_pi'])
    prod_PLCb_DA  = f_prod_PLCb_DA(p['v_DA'], p['d_rest'], p['beta'], p['K_DA'], p['K_p'], p['c_rest'], p['K_pi'])
    prod_PLCd     = f_prod_PLCd(p['v_delta'], i, p['kappa_delta'], p['c_rest'], p['K_PLCdelta'])
    degr_IP3_3K   = f_degr_IP3_3K(p['v_3K'], p['c_rest'], p['K_D'], i, p['K_3'])
    degr_IP_5P    = f_degr_IP_5P(p['r_5P'], i)
    
    return prod_PLCb_glu + prod_PLCd + prod_PLCb_DA - degr_IP3_3K - degr_IP_5P


# Auxiliary

@njit
def calculate_stimuli(neut = (0,), stimuli_times = (0,), t = 0, dt = 1, stimuli_types = (0,), stimuli_t_init = (0,), 
                      stimuli_t_end = (0,), stimuli_Hz = (0,), stimuli_comparts = (0,), rho = (0,)):
    """Simulate the release of neurotransmitter from a pre-synaptic neuron. 
    
    The mode of neurotransmitter release is defined by stimulus type. If the stimulus
    type is poissonian (stimuli_types = "poisson"), the pre-synaptic spike times 
    (neurotransmitter release) are drawn from a poisson (exponential time) and the 
    neurotransmitter concentration in each compartment is incremented by rho. The 
    frequency of the poisson distribution is given by 1/Hz (1/stimuli_Hz). If the 
    stimulus type is constant, the concentration is set to rho.

    Stimuli are applied from stimuli_t_init to stimuli_t_end

    Parameters
    ----------
    neut: list or numpy 1D-array 
        neurotransmition concentration and length equal the number of compartments 
        under stimulation.
    stimuli_times: list or numpy 1D-array
        previous pre-synaptic spike times for each compartment under stimulation.
    t: float
        current time.
    dt: float
        time step in the integration method.
    stimuli_types: list or tuple
        type of stimuli - "poisson", "constant" or none. If neither option were given,
        the neurotransmitter value will be set to zero.
    stimuli_t_init: list, tuple or numpy 1D-array
        initial stimulation time for each compartment under stimulation.
    stimuli_t_end: list, tuple or numpy 1D-array 
        ending stimulation time for each compartment under stimulation.
    stimuli_Hz: list, tuple or numpy 1D-array 
        frequency (in seconds) for each compartment under stimulation.
    stimuli_comparts: list, tuple or numpy 1D-array 
        compartments under stimulation.
    rho: float
        if stimulus type is "poisson", it is the increment that the neurotransmitter 
        receive for each release event. If stimulus_type is "constant", it is the value
        at which the neurotransmitter concentrations is fixed.

    Return
    ----------
    neut: list or numpy 1D-array
        updated neurotransmitter concentrations. Its length equals the number of 
        compartments under stimulation.
    stimuli_times: list or numpy 1D-array
        updated pre-synaptic spike times for poissonian stimulation.
    """

    n_stim = len(stimuli_types)
    for i_stim in range(n_stim):

        stimulus_type  = stimuli_types[i_stim]

        if stimulus_type == 'poisson':
        
            stimulus_compartments = stimuli_comparts[i_stim]
            n_stim_comparts = len(stimulus_compartments)

            Hz = stimuli_Hz[i_stim]
            
            for i_compart in range(n_stim_comparts):

                compart = int(stimulus_compartments[i_compart])
                t_init = stimuli_t_init[i_stim]
                t_end = stimuli_t_end[i_stim]
                
                if t == 0:
                    stimuli_times[i_stim, compart-1] += np.ceil(np.random.exponential(scale=1/Hz,size=1)/dt)[0] + int(t_init/dt)

                elif (t == stimuli_times[i_stim, compart-1]) & (t >= t_init/dt) & (t <= t_end/dt):
                    neut[compart-1] = neut[compart-1] + rho
                    stimuli_times[i_stim, compart-1] += np.ceil(np.random.exponential(scale=1/Hz,size=1)/dt)[0]
                                        
        elif stimulus_type == 'constant':
                
            stimulus_compartments = stimuli_comparts[i_stim]
            n_stim_comparts = len(stimulus_compartments)
                    
            for i_compart in range(n_stim_comparts):
                
                compart = int(stimulus_compartments[i_compart])
                t_init = stimuli_t_init[i_stim]
                t_end = stimuli_t_end[i_stim]

                if (t >= t_init/dt) & (t <= t_end/dt):
                    neut[compart-1] = rho
                else:
                    neut[compart-1] = 0
        else:
            for ic in range(len(neut)): neut[ic] = 0
    
    return neut, stimuli_times

@njit
def solve_model_equations(dt = 0.01, sample_rate = 100, compartment_to_monitor = (0), t_total = None, n_comparts = None, 
                          connection_matrix = None, parameters = None, A = None, Vol = None, r_ER = None, stim_glu_types = None, 
                          stim_glu_t_init = None, stim_glu_t_end = None, stim_glu_Hz = None, stim_glu_comparts = None, stim_DA_types = None, 
                          stim_DA_t_init = None, stim_DA_t_end = None, stim_DA_Hz = None, stim_DA_comparts = None,
                          Ca_ERalpha = None):
    """Calculate the nummerical solution of the system of differential equations of
    the simplified astrocyte model with the 4th-order Runge-Kutta method.

    Parameters
    ----------
    dt: float
        time step for the nummerical solution by the 4th-order Runge-Kutta method.
    sample_rate: integer
        sample rate for the intracellular calcium concentration output.
    compartment_to_monitor: list, tuple or numpy 1D-array 
        indicates which compartments to monitor the intracellular calcium 
        concentration.
    t_total: integer
        total simulation time (in seconds).
    n_comparts: integer
        number of compartments in the astrocyte compartmental model.
    connection_matrix: numpy 2D-array 
        compartment connections.
    parameters: numba dictionary 
        all model parameters.
    A: list, tuple or numpy 1D-array
        area of each compartment.
    Vol: list, tuple or numpy 1D-array 
        volume of each compartment
    r_ER: list, tuple or numpy 1D-array 
        cytosol-ER volume ratio of each compartment
    stim_glu_types: list or tuple 
        type of stimuli ("poisson", "constant" or "none") for the glutamatergic input.
    stim_glu_t_init: list, tuple or numpy 1D-array 
        initial time of glutamatergic stimulation.
    stim_glu_t_end: list, tuple or numpy 1D-array
        end time of glutamatergic stimulation.
    stim_glu_Hz: list, tuple or numpy 1D-array 
        frequency (in seconds) of glutamatergic input.
    stim_glu_comparts: list, tuple or numpy 1D-array 
        compartments under stimulation with the glutamatergic input.
    stim_DA_types: list or tuple
        type of stimuli ("poisson", "constant" or "none") for the dopaminergic input
    stim_DA_t_init: list, tuple or numpy 1D-array.
        initial time of dopaminergic stimulation 
    stim_DA_t_end: ist, tuple or numpy 1D-array 
        end time of dopaminergic stimulation.
    stim_DA_Hz: list, tuple or numpy 1D-array
        frequency (in seconds) of dopaminergic input.
    stim_DA_comparts: list, tuple or numpy 1D-array
        compartments under stimulation for the dopaminergic input.
    Ca_ERalpha: numpy 1D-array
        alpha factor multiplying the Ca_ER concentration by the intracellular Ca, ensures equilibrium without input
    Return
    ------
    Ca_out: numpy array
        intracellular calcium concentration of the compartments given by the parameter
        compartment_to_monitor and with sample rate given by sample_rate
    """                
    n_points = int(t_total/dt)
    
    p = parameters 
    M = connection_matrix
        
    # Model Output
    c_out = np.ones(shape=(len(compartment_to_monitor), int(n_points/sample_rate)))*p['c_rest']
    i_out = np.ones(shape=(len(compartment_to_monitor), int(n_points/sample_rate)))*p['i_rest']

    # Initial Values
    c = np.ones(shape=(n_comparts))*p['c_rest']
    i = np.ones(shape=(n_comparts))*p['i_rest']
    g = np.ones(shape=(n_comparts))*p['g_rest']
    d = np.ones(shape=(n_comparts))*p['d_rest']

    # Stimuli
    stimuli_times_glu = np.zeros(shape = (len(stim_glu_types), n_comparts))
    stimuli_times_DA = np.zeros(shape = (len(stim_DA_types), n_comparts))

    ### Runge Kutta 4th Order ###
    for i_t in range(n_points):

        # Stimuli
        g, stimuli_times_glu = calculate_stimuli(g, stimuli_times_glu, i_t, dt, stim_glu_types, stim_glu_t_init, stim_glu_t_end,
                                                    stim_glu_Hz, stim_glu_comparts, p['rho_glu'])
        d, stimuli_times_DA  = calculate_stimuli(d, stimuli_times_DA, i_t, dt, stim_DA_types, stim_DA_t_init, stim_DA_t_end, 
                                                    stim_DA_Hz, stim_DA_comparts, p['rho_DA'])

        # Runge-Kutta
        k1 = dt * model_eqs(M, p, A, Vol, r_ER, Ca_ERalpha, c, i, g, d)
        
        k2 = dt * model_eqs(M, p, A, Vol, r_ER, Ca_ERalpha, c + 0.5*k1[0], i + 0.5*k1[1], 
                            g + 0.5*k1[2], d + 0.5*k1[3])
        
        k3 = dt * model_eqs(M, p, A, Vol, r_ER, Ca_ERalpha, c + 0.5*k2[0], i + 0.5*k2[1], 
                            g + 0.5*k2[2], d + 0.5*k2[3])
        
        k4 = dt * model_eqs(M, p, A, Vol, r_ER, Ca_ERalpha, c + k3[0], i + k3[1],
                            g + k3[2], d + k3[3])
        
        c += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
        i += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6
        g += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6
        d += (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6
        
        if i_t % sample_rate == 0:
            for ic, compart in enumerate(compartment_to_monitor):
                c_out[ic, int(i_t/sample_rate)] = c[compart - 1]
                i_out[ic, int(i_t/sample_rate)] = i[compart - 1]
        
    return c_out, i_out


# Calculate stationary values for initializing variables and alpha Ca_ER


def null_IP3(i, p):

    alpha = p['v_delta']*p['kappa_delta'] * (p['c_rest']**2 / (p['c_rest']**2 + p['K_PLCdelta']**2))
    beta = p['v_3K'] * (p['c_rest']**4 / (p['c_rest']**4 + p['K_D']**4))

    return alpha/(p['kappa_delta'] + i) - beta*(i / (p['K_3'] + i)) - p['r_5P']*i

def null_caer(alpha, p, A, Vol, r_ER):
    
    c = p['c_rest']
    i = p['i_rest']
    
    h = p['h']**3

    a = quartic(r_ER, -2.61407715e+02, 1.10942956e+02, -1.59939929e+01, 8.92587873e-01, 1.39080128e-02)*1369.0
    b = -1.03006013e-04/r_ER + 1.00099667e+00
        
    Ca_ER = alpha*(a - b/r_ER*c)
    
    I_NCX     = f_I_NCX(p['alpha_NCX'], p['beta_NCX'], c)
    I_IP3R    = p['F']* Vol / A * p['r_C'] * h * (i/(i+p['d_1']))**3 * (c/(c+p['d_5']))**3 * (Ca_ER-c)
    I_SERCA   = f_I_SERCA(p['F'], A, Vol, p['v_ER'], c, p['K_ER'])
    I_CERleak = p['F'] * Vol / A * p['r_L'] * (Ca_ER - c)

    return A/(Vol*p['F'])*I_NCX + A*np.sqrt(r_ER)/(Vol*p['F'])*(I_IP3R - I_SERCA + I_CERleak)
    
