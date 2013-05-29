import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import * 
from SimGeneral.MixingModule.mixPoolSource_cfi import * 

mix = cms.EDProducer("MixingModule",
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
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55),
          probValue = cms.vdouble(
                2.54656e-11,
                6.70051e-11,
                2.74201e-06,
                6.9111e-06,
                5.00919e-06,
                6.24538e-05,
                0.000338679,
                0.000892795,
                0.00237358,
                0.00686023,
                0.0144954,
                0.026012,
                0.0360377,
                0.0420151,
                0.0457901,
                0.0482319,
                0.0503176,
                0.052569,
                0.0546253,
                0.0561205,
                0.0568903,
                0.0570889,
                0.0566598,
                0.0553747,
                0.0531916,
                0.0501454,
                0.0463101,
                0.0417466,
                0.0364842,
                0.0306443,
                0.0245417,
                0.0186276,
                0.0133446,
                0.00900314,
                0.00571947,
                0.00342706,
                0.00194292,
                0.00104671,
                0.000538823,
                0.000266973,
                0.000128572,
                6.09778e-05,
                2.89549e-05,
                1.40233e-05,
                7.04619e-06,
                3.71289e-06,
                2.055e-06,
                1.18713e-06,
                7.08603e-07,
                4.32721e-07,
                2.6817e-07,
                1.67619e-07,
                1.05157e-07,
                6.59446e-08,
                4.11915e-08,
                2.55494e-08
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



