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
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 ns

    bunchspace = cms.int32(1),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    mixObjects = cms.PSet(
        mixHepMC = cms.PSet(
            input = cms.VInputTag(cms.InputTag("generator")),
            makeCrossingFrame = cms.untracked.bool(True),
            type = cms.string('HepMCProduct')
            )
        ),

)

mix.mixObjects.mixHepMC.makeCrossingFrame = True



