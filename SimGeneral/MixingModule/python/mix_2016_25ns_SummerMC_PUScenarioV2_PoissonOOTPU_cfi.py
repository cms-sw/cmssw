import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import theMixObjects
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.digitizers_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-12), ## in terms of 25 nsec

    bunchspace = cms.int32(25), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74),
	  probValue = cms.vdouble(
			2.5e-06,
			7.5e-06,
			3.25e-05,
			5.4e-05,		
			6.95e-05,
			9.45e-05,
			0.000158,
			0.000481,
			0.000935,
			0.001701,
			0.0035815,
			0.0066195,
			0.0104665,
			0.0147045,
			0.019025,
			0.024367,
			0.029137,
			0.0339585,
			0.038155,
			0.042018,
			0.045622,
			0.0482715,
		       	0.0487025,
			0.048112,
			0.046113,
			0.0438875,
			0.040798,
			0.0376485,
			0.0346425,
			0.031234,
			0.0283995,
			0.026178,
			0.023713,
			0.0215295,
			0.020064,
			0.018726,
			0.017375,
			0.0163485,
			0.0156265,
			0.0148855,
			0.014164,
			0.0135625,
			0.0129695,
			0.012526,
			0.0119585,
			0.011231,
			0.0103575,
			0.0095385,
			0.008675,
			0.0078355,
			0.0069075,
			0.00582,
			0.0049585,
			0.0040605,
			0.00317,
			0.002581,
			0.0018915,
			0.001432,
			0.000993,
			0.0006825,
			0.0004615,
			0.000297,
			0.000195,
			0.0001155,
			8.2e-05,
			4.2e-05,
			2.35e-05,
			1.25e-05,
			6e-06,
			2e-06,
			2e-06,
			1.5e-06,
			0.0,
			0.0,
			0.0
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



