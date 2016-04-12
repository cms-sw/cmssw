import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_cfi import *

# esmodule creating  records + corresponding empty essource
from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_mc_cff import *


#Common
def _modifyecalTriggerPrimitiveDigis_with_suppressedCommon( obj ):
    obj.BarrelOnly = cms.bool(True)  

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( simEcalTriggerPrimitiveDigis, func=_modifyecalTriggerPrimitiveDigis_with_suppressedCommon )
e
