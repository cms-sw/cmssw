import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi import *


# esmodule creating  records + corresponding empty essource
# when commented, one takes the configuration from the global tag
#
#from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *

#Common
def _modifyecalTriggerPrimitiveDigisCommon( obj ):
    obj.BarrelOnly = cms.bool(True)  

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( simEcalTriggerPrimitiveDigis, func=_modifyecalTriggerPrimitiveDigisCommon )
