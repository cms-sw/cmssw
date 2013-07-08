import FWCore.ParameterSet.Config as cms

# configuration to model pileup based on full 2011 dataset
from SimGeneral.MixingModule.mixObjects_cfi import * 
from SimGeneral.MixingModule.mixPoolSource_cfi import * 

mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-12),  ## in terms of 25 nsec

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34),
          probValue = cms.vdouble(
                               1.30976e-05,
                               0.000148266,
                               0.00226073,
                               0.030543,
                               0.0868303,
                               0.120295,
                               0.124687,
                               0.110419,
                               0.0945742,
                               0.0837875,
                               0.0774277,
                               0.0740595,
                               0.0676844,
                               0.0551203,
                               0.0378357,
                               0.0210203,
                               0.00918262,
                               0.00309786,
                               0.000808509,
                               0.000168568,
                               3.02344e-05,
                               5.16455e-06,
                               8.83185e-07,
                               1.43975e-07,
                               2.07228e-08,
                               2.51393e-09,
                               2.52072e-10,
                               2.07328e-11,
                               1.39369e-12,
                               7.63843e-14,
                               3.4069e-15,
                               1.23492e-16,
                               3.63059e-18,
                               8.53277e-20,
                               1.33668e-22 ),
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
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)



