import FWCore.ParameterSet.Config as cms

# configuration to model pileup for Chamonix 2012 studies
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-12), ## in terms of 25 nsec

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50),
          probValue = cms.vdouble(
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608,
                       0.019608
                    ),

          histoFileName = cms.untracked.string('histProbFunction.root'),
        ),
	sequential = cms.untracked.bool(False),
        manage_OOT = cms.untracked.bool(True),  ## manage out-of-time pileup
        ## setting this to True means that the out-of-time pileup
        ## will have a different distribution than in-time, given
        ## by what is described on the next line:
        OOT_type = cms.untracked.string('Poisson'),  ## generate OOT with a Poisson matching the number chosen for in-time
        #OOT_type = cms.untracked.string('fixed'),  ## generate OOT with a fixed distribution
        #intFixed_OOT = cms.untracked.int32(2),
        fileNames = FileNames
    ),
    mixObjects = cms.PSet(theMixObjects)
)



