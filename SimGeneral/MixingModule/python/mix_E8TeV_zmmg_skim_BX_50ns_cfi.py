import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-12), ## in 50ns spacing, go 150ns into past

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
             probFunctionVariable = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55),
             probValue = cms.vdouble(
                       2.22595e-08,
                       2.00205e-07,
                       1.64416e-06,
                       1.14116e-05,
                       6.40739e-05,
                       0.000284833,
                       0.000994025,
                       0.00272863,
                       0.00613133,
                       0.0135242,
                       0.0322459,
                       0.0601265,
                       0.0778966,
                       0.0818541,
                       0.0798978,
                       0.0752475,
                       0.0687418,
                       0.0636304,
                       0.0604193,
                       0.0571488,
                       0.0537445,
                       0.050468,
                       0.0470076,
                       0.04297,
                       0.0377791,
                       0.0310346,
                       0.0231801,
                       0.0154523,
                       0.00910777,
                       0.00473839,
                       0.00218447,
                       0.000898356,
                       0.000331497,
                       0.000110006,
                       3.27802e-05,
                       8.75341e-06,
                       2.1002e-06,
                       4.58968e-07,
                       9.41886e-08,
                       1.90252e-08,
                       3.95869e-09,
                       8.60669e-10,
                       1.91046e-10,
                       4.18017e-11,
                       8.78577e-12,
                       1.74995e-12,
                       3.28416e-13,
                       5.79467e-14,
                       9.60488e-15,
                       1.49505e-15,
                       2.18539e-16,
                       2.99303e-17,
                       3.87579e-18,
                       4.53878e-19,
                       3.6847e-20,
                       0),
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


