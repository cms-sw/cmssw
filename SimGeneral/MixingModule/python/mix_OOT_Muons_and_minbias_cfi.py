import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(4),
    minBunch = cms.int32(-4), ## in terms of 25 nsec

    bunchspace = cms.int32(25), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('fixed'),
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(40.0)
        ),

	sequential = cms.untracked.bool(False),
        manage_OOT = cms.untracked.bool(True),  ## manage out-of-time pileup
        ## setting this to True means that the out-of-time pileup
        ## will have a different distribution than in-time, given
        ## by what is described on the next line:
        #OOT_type = cms.untracked.string('Poisson'),  ## generate OOT with a Poisson matching the number chosen for in-time
        OOT_type = cms.untracked.string('fixed'),  ## generate OOT with a fixed distribution
        intFixed_OOT = cms.untracked.int32(0), ## we only want in-time pileup, but muons
                                               ## need out of time bunch crossings
        fileNames = FileNames
    ),
    cosmics = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.)
        ),
        type = cms.string('fixed'),
        sequential = cms.untracked.bool(False),
        maxBunch_cosmics = cms.untracked.int32(4),
        minBunch_cosmics = cms.untracked.int32(4),
        fileNames = cms.untracked.vstring('file:SingleMuPt2_50_cfi_GEN_SIM.root')
    ),
    mixObjects = cms.PSet(theMixObjects)
)



