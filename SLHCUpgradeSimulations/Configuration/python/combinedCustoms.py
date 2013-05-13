from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import customise as customiseBE
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import customise as customiseLB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import customise as customiseLB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import l1EventContent as customise_ev_BE
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import l1EventContent as customise_ev_LB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import l1EventContent as customise_ev_LB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk

def cust_phase2_BE(process):
    process=customisePostLS1(process)
    process=customiseBE(process)
    process=customise_ev_BE(process)
    return process

def cust_phase2_LB6PS(process):
    process=customisePostLS1(process)
    process=customiseLB6PS(process)
    process=customise_ev_LB6PS(process)
    return process

def cust_phase2_LB4LPS_2L2S(process):
    process=customisePostLS1(process)
    process=customiseLB4LPS_2L2S(process)
    process=customise_ev_LB4LPS_2L2S(process)
    return process

def cust_2017(process):
    process=customisePostLS1(process)
    process=customise_Phase1Tk(process)

    return process
    
def noCrossing(process):
    process=customise_NoCrossing(process)
    return process





    
