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
		0.0,
		2.25e-06,
		1.4e-05,
		2.6e-05,
		3.925e-05,
		5.175e-05,
		6.7e-05,
		0.00015475,
		0.0005675,
		0.00133025,
		0.00265725,
		0.004463,
		0.00687675,
		0.00937025,
		0.01185625,
		0.01443875,
		0.01759725,
		0.021601,
		0.0265645,
		0.032385,
		0.037999,
		0.042104,
		0.043781,
		0.0437285,
		0.0421535,
		0.03999025,
		0.037566,
		0.03511325,
		0.0326895,
		0.030577,
		0.02892425,
		0.027651,
		0.0263845,
		0.02493575,
		0.0246165,
		0.02377375,
		0.02326225,
		0.02241775,
		0.022166,
		0.02147225,
		0.02086025,
		0.02001575,
		0.019238,
		0.0187265,
		0.017941,
		0.01683225,
		0.0155065,
		0.01433175,
		0.012936,
		0.01185675,
		0.010332,
		0.00873225,
		0.007452,
		0.00598575,
		0.0047895,
		0.00389475,
		0.00277425,
		0.00214125,
		0.00147,
		0.00099525,
		0.0006795,
		0.00044025,
		0.000279,
		0.000168,
		0.000123,
		5.85e-05,
		3.525e-05,
     		1.875e-05,
		9e-06,
		6e-06,
		0.0,
		0.0,
		2.25e-06,
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



