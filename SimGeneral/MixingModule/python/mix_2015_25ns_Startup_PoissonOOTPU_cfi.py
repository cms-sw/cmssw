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
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36),
          probValue = cms.vdouble(
                      1.57E-06,
                      4.62E-06,
                      1.26E-05,
                      4.71E-05,
                      1.41E-04,
                      6.29E-04,
                      7.16E-03,
                      2.40E-02,
                      4.12E-02,
                      6.50E-02,
                      9.36E-02,
                      1.18E-01,
                      1.27E-01,
                      1.11E-01,
                      8.88E-02,
                      7.19E-02,
                      5.95E-02,
                      4.89E-02,
                      3.90E-02,
                      3.01E-02,
                      2.26E-02,
                      1.65E-02,
                      1.18E-02,
                      8.21E-03,
                      5.53E-03,
                      3.61E-03,
                      2.28E-03,
                      1.39E-03,
                      8.17E-04,
                      4.63E-04,
                      2.53E-04,
                      1.33E-04,
                      6.75E-05,
                      3.29E-05,
                      1.57E-05,
                      7.85E-06,
                      3.14E-06),
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



