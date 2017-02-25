import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# simHcalUnsuppressedDigis is now done inside mixing module
hcalDigiSequence = cms.Sequence(simHcalTriggerPrimitiveDigis
                                +simHcalDigis
                                *simHcalTTPDigis)

#_phase2_hcalDigiSequence = hcalDigiSequence.copyAndExclude([simHcalTriggerPrimitiveDigis,simHcalTTPDigis])
#_phase2_hcalDigiSequence = hcalDigiSequence.copyAndExclude([simHcalTTPDigis])

#from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
#phase2_hcal.toReplaceWith( hcalDigiSequence, _phase2_hcalDigiSequence )
