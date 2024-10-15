###############################################################################
#
#
#
###############################################################################

def define_parameters():
    """Create a Python dictionaruy and return all model's parameters."""

    p = {}
    
    ### Physical Constants ###
    p['F'] = 96500               # C/mole

    ### Resting State ###
    p['c_rest']  = 0.1         
    p['i_rest']  = 0.1    
    p['g_rest'] = 0              
    p['d_rest']  = 0
    p['h'] = 0.8

    ### IP3 Dynamics ###
    # PLC B Synthesis
    p['K_p']  = 0.08        
    p['K_pi'] = 0.8214          
    
    # IP3 PLC Delta Synthesis
    p['v_delta']     = 0.013        # 1/s
    p['kappa_delta'] = 0.782    
    p['K_PLCdelta']  = 0.1369
    
    # IP3-3K Degradation
    p['v_3K']        = 1.043        # 1/s 
    p['K_D']         = 0.958   
    p['K_3']         = 0.522
       
    # IP-5P Degradation
    p['r_5P']    = 0.04             # 1/s


    ### Glutamate Transmission ###
    p['rho_glu']    = 0.5e-3
    p['G_glu']      = 100           # 1/s    
    p['K_R']        = 0.00104
    p['v_beta']     = 0.211         # 1/s
    p['alpha']      = 0.7
        
    ### Dopamine Transmission ###
    p['rho_DA']     = 1e-3
    p['G_DA']       = 4.201         # 1/s    
    p['v_DA']       = 0.013         # 1/s
    p['K_DA']       = 5e-3     
    p['beta']       = 0.5
       
    ### SERCA Current ###
    p['v_ER'] = 18.7818             # 1/s
    p['K_ER'] = 0.116365
        
    ### Ca ER Leak Current ###
    p['r_L']  = 0.11                # 1/s       
    
    ### h Dynamics ###
    p['d_1'] = 0.0646  
    p['d_5'] = 0.10735
    p['r_C'] = 6.0                  # 1/s     
    
    ### NCX ###
    p['alpha_NCX'] = 1.169e-4       # A/mM.m2
    p['beta_NCX']  = 1.23177478e-05 # A/mM.m2
    
    
    ### Diffusion Constants ###
    p['D_Ca']   = 0.3          # 1/s
    p['D_IP3']  = 0.3          # 1/s
    p['D_glu']  = 4e-4         # 1/s
    p['D_DA']   = 13.8         # 1/s
    
    return p