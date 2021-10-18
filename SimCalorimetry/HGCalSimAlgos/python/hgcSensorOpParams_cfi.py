import FWCore.ParameterSet.Config as cms

def hgcSiSensorIleak(version):

    """ 
    this method returns different parameterizations of the leakage current for different versions
    TDR_{600V,800V} - TDR based parameterizations for operations at -30C and 600V or 800V
    CERN21_{600V,800V}_{annealing} - 2021 CERN-based parameterizations for operations at -30C, 600V or 800V and different annealing times
    if version is unknown a ValueError exception is raised
    """
    
    if version=='TDR_600V':
        return [0.993,-42.668]
    elif version=='TDR_800V':
        return [0.996,-42.464]
    elif version=='CERN21_600V_10m':
        return [1.0,-42.457]
    elif version=='CERN21_600V_30m':
        return [1.0,-42.569]
    elif version=='CERN21_600V_90m':
        return [1.0,-42.715]
    elif version=='CERN21_600V_120m':
        return [1.0,-42.741]
    elif version=='CERN21_800V_10m':
        return [1.0,-42.267]
    elif version=='CERN21_800V_30m':
        return [1.0,-42.369]
    elif version=='CERN21_800V_90m':
        return [1.0,-42.509]
    elif version=='CERN21_800V_120m':
        return [1.0,-42.530]

    raise ValueError('version={} is unknown to retrieve Ileak parameterization for HGC Si sensors'.format(version))


def hgcSiSensorCCE(sensor,version):

    """ 
    this method returns different parameterizations of the charge collection efficiency (CCE)
    for different sensor types (sensor) and measurement versions (version)
    sensor = 120,200,300
    version = TDR_{600V,800V}   - TDR based measurements at different voltages
              TTU_{600V,800V}   - Texas Tech based measurements at different voltages
              CERN21_{600V,800V}_{annealing} -  CERN2021 based measurements at different voltages and annealing times
    if the pair (sensor,version) is unknown a ValueError exception is raised
    """
    
    if version=='TDR_600V':
        if sensor==120  : return [3.5e+15,0.00675,0.6381]
        elif sensor==200: return [1e+15,-0.2749,10.17]
        elif sensor==300: return [3e+14,-0.2823,10.16]
    elif version=='TDR_800V':
        if sensor==120  : return [3.5e+15,0.008824,0.6342]
        elif sensor==200: return [1.5e+15,-0.2705,10.15]
        elif sensor==300: return [6e+14,-0.3535,12.71]
    elif version=='TTU_600V':
        if sensor==120  : return [1.5e+15,  9.98631e-18, 0.343774]
        elif sensor==200: return [1.5e+15, -2.17083e-16, 0.304873]
        elif sensor==300: return [6e+14,   -8.01557e-16, 0.157375]
    elif version=='TTU_800V':
        if sensor==120  : return [1.5e+15, 3.35246e-17,  0.251679]
        elif sensor==200: return [1.5e+15, -1.62096e-16, 0.293828]
        elif sensor==300: return [6e+14,   -5.95259e-16, 0.183929]
    elif version=='CERN21_600V_10m':
        if sensor==120  : return [1.35e+15,-0.2452,9.591]
        elif sensor==200: return [9e+14,-0.3186,11.95]
        elif sensor==300: return [5.85e+14,-0.2668,9.701]
    elif version=='CERN21_600V_30m':
        if sensor==120  : return [1.35e+15,-0.2105,8.362]
        elif sensor==200: return [9e+14,-0.4191,15.48]
        elif sensor==300: return [5.85e+14,-0.2699,9.89]
    elif version=='CERN21_600V_90m':
        if sensor==120  : return [1.35e+15,-0.1954,7.769]
        elif sensor==200: return [9e+14,-0.2354,8.983]
        elif sensor==300: return [5.85e+14,-0.2377,8.79]
    elif version=='CERN21_600V_120m':
        if sensor==120  : return [1.35e+15,-0.1775,7.119]
        elif sensor==200: return [9e+14,-0.2257,8.647]
        elif sensor==300: return [5.85e+14,-0.2544,9.369]
    elif version=='CERN21_800V_10m':
        if sensor==120  : return [1.35e+15,-0.2031,8.148]
        elif sensor==200: return [9e+14,-0.1833,7.32]
        elif sensor==300: return [5.85e+14,-0.3131,11.45]
    elif version=='CERN21_800V_30m':
        if sensor==120  : return [1.35e+15,-0.1731,7.097]
        elif sensor==200: return [9e+14,-0.3653,13.68]
        elif sensor==300: return [5.85e+14,-0.269,10]
    elif version=='CERN21_800V_90m':
        if sensor==120  : return [1.35e+15,-0.155,6.387]
        elif sensor==200: return [9e+14,-0.198,7.739]
        elif sensor==300: return [5.85e+14,-0.2023,7.701]
    elif version=='CERN21_800V_120m':
        if sensor==120  : return [1.35e+15,-0.1443,5.997]
        elif sensor==200: return [9e+14,-0.1821,7.172]
        elif sensor==300: return [5.85e+14,-0.2068,7.855]

    raise ValueError('sensor={} version={} is unknown to retrieve CCE parameterization for HGC Si sensors'.format(sensor,version))
