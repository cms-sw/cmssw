import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# simHcalUnsuppressedDigis is now done inside mixing module
hcalDigiSequence = cms.Sequence(simHcalTriggerPrimitiveDigis
                                +simHcalDigis
                                *simHcalTTPDigis)

def _modifySimCalorimetryHcalDigiSequenceForCommon( obj ):
    obj.remove(simHcalTriggerPrimitiveDigis)
    obj.remove(simHcalTTPDigis)

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( hcalDigiSequence, func=_modifySimCalorimetryHcalDigiSequenceForCommon )
