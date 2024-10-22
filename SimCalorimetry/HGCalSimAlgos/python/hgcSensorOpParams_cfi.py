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
        if sensor==120  : return [3.5e+15,10.31,-0.2635]
        elif sensor==200: return [9e+14,8.99,-0.241]
        elif sensor==300: return [3e+14,10.16,-0.2823]
    elif version=='TDR_800V':
        if sensor==120  : return [3.5e+15,10.39,-0.2638]
        elif sensor==200: return [1.5e+15,10.41,-0.2779]
        elif sensor==300: return [5e+14,12.59,-0.3501]
    elif version=='CERN21_600V_10m':
        if sensor==120  : return [1.35e+15,9.591,-0.2452]
        elif sensor==200: return [9e+14,11.95,-0.3186]
        elif sensor==300: return [5.85e+14,9.701,-0.2668]
    elif version=='CERN21_600V_30m':
        if sensor==120  : return [1.35e+15,8.362,-0.2105]
        elif sensor==200: return [9e+14,15.48,-0.4191 ]
        elif sensor==300: return [5.85e+14,9.89,-0.2699]
    elif version=='CERN21_600V_90m':
        if sensor==120  : return [1.35e+15,7.769,-0.1954]
        elif sensor==200: return [9e+14,8.983,-0.2354]
        elif sensor==300: return [5.85e+14,8.79,-0.2377]
    elif version=='CERN21_600V_120m':
        if sensor==120  : return [1.35e+15,7.119,-0.1775]
        elif sensor==200: return [9e+14,8.647,-0.2257 ]
        elif sensor==300: return [5.85e+14,9.369,-0.2544]
    elif version=='CERN21_800V_10m':
        if sensor==120  : return [1.35e+15,8.148,-0.2031]
        elif sensor==200: return [9e+14,7.32,-0.1833]
        elif sensor==300: return [5.85e+14,11.45,-0.3131]
    elif version=='CERN21_800V_30m':
        if sensor==120  : return [1.35e+15,7.097,-0.1731]
        elif sensor==200: return [9e+14,13.68,-0.3653]
        elif sensor==300: return [5.85e+14, 10,-0.269]
    elif version=='CERN21_800V_90m':
        if sensor==120  : return [1.35e+15,6.387,-0.155]
        elif sensor==200: return [9e+14,7.739,-0.198]
        elif sensor==300: return [5.85e+14,7.701,-0.2023]
    elif version=='CERN21_800V_120m':
        if sensor==120  : return [1.35e+15,5.997,-0.1443]
        elif sensor==200: return [9e+14,7.172,-0.1821]
        elif sensor==300: return [5.85e+14,7.855,-0.2068]

    raise ValueError('sensor={} version={} is unknown to retrieve CCE parameterization for HGC Si sensors'.format(sensor,version))
