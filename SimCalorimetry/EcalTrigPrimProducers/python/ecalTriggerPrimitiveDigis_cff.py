import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi import *


# esmodule creating  records + corresponding empty essource
# when commented, one takes the configuration from the global tag
#
#from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *

#Common
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( simEcalTriggerPrimitiveDigis, BarrelOnly = cms.bool(True) )
