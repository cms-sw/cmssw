import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise_Digi_TTI
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent_TTI as customise_ev_BE5D
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase0


def cust_phase2_BE5D(process):
    
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    	# additional customisation for digitisation_step :
    process = customise_Digi_TTI(process)
    process = customise_HcalPhase0(process)
    process = customise_ev_BE5D(process)
    
    return process


