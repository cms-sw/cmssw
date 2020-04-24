import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup for final scheme
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(50), ## ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(75.3), #The Xsec is in mb
            Lumi = cms.double(0.13) # The lumi is in E33 cm-2s-1
        ),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = FileNames
      ),
    mixObjects = cms.PSet(theMixObjects)
)


