import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitivePhase2Digis_cfi import *


# esmodule creating  records + corresponding empty essource
# when commented, one takes the configuration from the global tag
#



from Configuration.Eras.Modifier_phase2_ecalTP_devel_cff import phase2_ecalTP_devel
phase2_ecalTP_devel.toModify( simEcalEBTriggerPrimitivePhase2Digis) 

