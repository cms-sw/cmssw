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

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52),
          probValue = cms.vdouble(
                                    4.71E-09,
                                    2.86E-06,
                                    4.85E-06,
                                    1.53E-05,
                                    3.14E-05,
                                    6.28E-05,
                                    1.26E-04,
                                    3.93E-04,
                                    1.42E-03,
                                    6.13E-03,
                                    1.40E-02,
                                    2.18E-02,
                                    2.94E-02,
                                    4.00E-02,
                                    5.31E-02,
                                    6.53E-02,
                                    7.64E-02,
                                    8.42E-02,
                                    8.43E-02,
                                    7.68E-02,
                                    6.64E-02,
                                    5.69E-02,
                                    4.94E-02,
                                    4.35E-02,
                                    3.84E-02,
                                    3.37E-02,
                                    2.92E-02,
                                    2.49E-02,
                                    2.10E-02,
                                    1.74E-02,
                                    1.43E-02,
                                    1.16E-02,
                                    9.33E-03,
                                    7.41E-03,
                                    5.81E-03,
                                    4.49E-03,
                                    3.43E-03,
                                    2.58E-03,
                                    1.91E-03,
                                    1.39E-03,
                                    1.00E-03,
                                    7.09E-04,
                                    4.93E-04,
                                    3.38E-04,
                                    2.28E-04,
                                    1.51E-04,
                                    9.83E-05,
                                    6.29E-05,
                                    3.96E-05,
                                    2.45E-05,
                                    1.49E-05,
                                    4.71E-06,
                                    2.36E-06),
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



