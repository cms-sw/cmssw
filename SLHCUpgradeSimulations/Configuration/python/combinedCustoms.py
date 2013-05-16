from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import customise as customiseBE
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import customise as customiseLB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import customise as customiseLB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import l1EventContent as customise_ev_BE
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import l1EventContent as customise_ev_LB6PS
from SLHCUpgradeSimulations.Configuration.phase2TkCustoms_LB_6PS import l1EventContent as customise_ev_LB4LPS_2L2S
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
import aging

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
    process=customisePhase1Tk(process)
    return process
    
def noCrossing(process):
    process=customise_NoCrossing(process)
    return process

def agePixel(process,lumi):
    process=agePixel(process,lumi)
    return process

def ageHcal(process,lumi):
    process=aging.ageHcal(process,lumi)
    return process

def ageEcal(process,lumi):
    process=aging.ageEcal(process,lumi)
    return process

def customise_aging_300(process):
    process=aging.customise_aging_300(process)
    return process

def customise_aging_500(process):
    process=aging.customise_aging_500(process)
    return process

def customise_aging_1000(process):
    process=aging.customise_aging_1000(process)
    return process

def customise_aging_3000(process):
    process=aging.customise_aging_3000(process)
    return process

def customise_aging_ecalonly_300(process):
    process=aging.customise_aging_ecalonly_300(process)
    return process

def customise_aging_ecalonly_1000(process):
    process=aging.customise_aging_ecalonly_1000(process)
    return process

def customise_aging_ecalonly_3000(process):
    process=aging.customise_aging_ecalonly_3000(process)
    return process

def customise_aging_newpixel_1000(process):
    process=aging.customise_aging_newpixel_1000(process)
    return process

def customise_aging_newpixel_3000(process):
    process=aging.customise_aging_newpixel_3000(process)
    return process

def ecal_complete_aging(process):
    process=aging.ecal_complete_aging(process)
    return process

def turn_off_HE_aging(process):
    process=aging.turn_off_HE_aging(process)
    return process

def turn_off_HF_aging(process):
    process=aging.turn_off_HF_aging(process)
    return process

def turn_off_Pixel_aging(process):
    process=aging.turn_off_Pixel_aging(process)
    return process

def turn_on_Pixel_aging_1000(process):
    process=aging.turn_on_Pixel_aging_1000(process)
    return process

def reco_aging_hcal_stdgeom(process):
    process=aging.reco_aging_hcal_stdgeom(process)
    return process

def reco_aging_hcal_stdgeom_300(process):
    process=aging.reco_aging_hcal_stdgeom_300(process)
    return process

def reco_aging_hcal_stdgeom_500(process):
    process=aging.reco_aging_hcal_stdgeom_500(process)
    return process

def reco_aging_hcal_stdgeom_1000(process):
    process=aging.reco_aging_hcal_stdgeom_1000(process)
    return process

def reco_aging_hcal_stdgeom_3000(process):
    process=aging.reco_aging_hcal_stdgeom_3000(process)
    return process
    
