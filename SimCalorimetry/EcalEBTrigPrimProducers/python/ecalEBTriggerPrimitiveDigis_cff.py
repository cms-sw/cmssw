import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cfi import *


# esmodule creating  records + corresponding empty essource
# when commented, one takes the configuration from the global tag
#


#Common
from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( simEcalEBTriggerPrimitiveDigis, BarrelOnly = cms.bool(True) )
