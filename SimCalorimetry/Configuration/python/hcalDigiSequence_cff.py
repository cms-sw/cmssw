import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# simHcalUnsuppressedDigis is now done inside mixing module
hcalDigiSequence = cms.Sequence(simHcalTriggerPrimitiveDigis
                                +simHcalDigis
                                *simHcalTTPDigis)

_phase2_hcalDigiSequence = hcalDigiSequence.copy()
_phase2_hcalDigiSequence.remove(simHcalTriggerPrimitiveDigis)
_phase2_hcalDigiSequence.remove(simHcalTTPDigis)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toReplaceWith( hcalDigiSequence, _phase2_hcalDigiSequence )
