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

    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42),
          probValue = cms.vdouble(
                     7.85E-07,
                     3.72E-06,
                     9.80E-06,
                     1.57E-05,
                     4.71E-05,
                     9.43E-05,
                     5.97E-04,
                     4.95E-03,
                     1.62E-02,
                     2.84E-02,
                     4.13E-02,
                     6.02E-02,
                     7.98E-02,
                     9.70E-02,
                     1.06E-01,
                     1.00E-01,
                     8.46E-02,
                     6.99E-02,
                     5.89E-02,
                     5.04E-02,
                     4.29E-02,
                     3.58E-02,
                     2.93E-02,
                     2.34E-02,
                     1.83E-02,
                     1.41E-02,
                     1.07E-02,
                     7.98E-03,
                     5.81E-03,
                     4.15E-03,
                     2.89E-03,
                     1.97E-03,
                     1.31E-03,
                     8.50E-04,
                     5.38E-04,
                     3.32E-04,
                     1.99E-04,
                     1.17E-04,
                     6.66E-05,
                     3.69E-05,
                     1.99E-05,
                     1.26E-06,
                     4.71E-07),
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



