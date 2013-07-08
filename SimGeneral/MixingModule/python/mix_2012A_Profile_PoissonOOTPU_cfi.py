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
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40),
          probValue = cms.vdouble(
                9.20919e-21,
                3.26111e-12,
                1.49952e-06,
                0.000871581,
                0.000790751,
                0.000128453,
                7.78403e-06,
                1.58194e-07,
                1.0553e-05,
                0.000139057,
                0.000302436,
                0.000441445,
                0.00351439,
                0.0167421,
                0.0382962,
                0.052312,
                0.0585257,
                0.0666345,
                0.079071,
                0.0933259,
                0.105656,
                0.11164,
                0.10704,
                0.0911808,
                0.0686565,
                0.0461453,
                0.0281433,
                0.0157818,
                0.0081649,
                0.0038717,
                0.0016626,
                0.000638312,
                0.000216668,
                6.44555e-05,
                1.66953e-05,
                3.74767e-06,
                7.26661e-07,
                1.21425e-07,
                1.74576e-08,
                2.15706e-09,
                2.28857e-10
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



