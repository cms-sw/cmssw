import FWCore.ParameterSet.Config as cms

# this is a minimum configuration of the Mixing module,
# to run it in the zero-pileup mode
#
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(450),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    mixObjects = cms.PSet(theMixObjects)
)

#
# Need to modify the pixel dynamic inefficiency if this is post LS1
#
from Configuration.StandardSequences.Eras import eras

# The modifyMixForPostLS1 function is defined in SimGeneral.MixingModule.digitizers_cfi
eras.postLS1.toModify( mix, func=modifyMixForPostLS1 )
# Also need to modify theDigitizersValid, because if validation is on
# process.mix.digitizers will be set to that later, which overwrites the
# above customisation.
from functools import partial
eras.postLS1.toModify( mix, func=partial(modifyMixForPostLS1,digitizers=theDigitizersValid) )
