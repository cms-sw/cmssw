import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cfi import *


# esmodule creating  records + corresponding empty essource
# when commented, one takes the configuration from the global tag
#


#Common
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( simEcalEBTriggerPrimitiveDigis, BarrelOnly = cms.bool(True) )
