import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import * 
from SimGeneral.MixingModule.mixPoolSource_cfi import * 

mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-2), ## in terms of 25 nsec

    bunchspace = cms.int32(50), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59),
          probValue = cms.vdouble(
                        2.620E-06,
                        5.400E-06,
                        1.488E-05,
                        5.006E-05,
                        1.001E-04,
                        1.502E-04,
                        2.157E-03,
                        6.738E-03,
                        1.120E-02,
                        1.428E-02,
                        1.736E-02,
                        2.126E-02,
                        2.649E-02,
                        3.370E-02,
                        4.318E-02,
                        5.271E-02,
                        5.875E-02,
                        6.032E-02,
                        5.807E-02,
                        5.372E-02,
                        4.984E-02,
                        4.742E-02,
                        4.578E-02,
                        4.414E-02,
                        4.213E-02,
                        3.971E-02,
                        3.694E-02,
                        3.391E-02,
                        3.074E-02,
                        2.750E-02,
                        2.430E-02,
                        2.119E-02,
                        1.824E-02,
                        1.550E-02,
                        1.300E-02,
                        1.075E-02,
                        8.776E-03,
                        7.066E-03,
                        5.611E-03,
                        4.394E-03,
                        3.393E-03,
                        2.583E-03,
                        1.938E-03,
                        1.434E-03,
                        1.046E-03,
                        7.520E-04,
                        5.329E-04,
                        3.722E-04,
                        2.563E-04,
                        1.739E-04,
                        1.164E-04,
                        7.677E-05,
                        4.993E-05,
                        3.203E-05,
                        2.027E-05,
                        1.266E-05,
                        7.798E-06,
                        4.743E-06,
                        2.849E-06,
                        1.201E-06),
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



