import FWCore.ParameterSet.Config as cms

# configuration to model pileup for full 2011 dataset
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
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49),
          probValue = cms.vdouble(2.30E-03,5.56E-03,1.28E-02,2.10E-02,2.75E-02,3.11E-02,3.19E-02,3.14E-02,3.09E-02,3.15E-02,3.34E-02,3.65E-02,4.04E-02,4.43E-02,4.80E-02,5.07E-02,5.26E-02,5.31E-02,5.25E-02,5.07E-02,4.81E-02,4.45E-02,4.03E-02,3.57E-02,3.09E-02,2.60E-02,2.14E-02,1.71E-02,1.34E-02,1.02E-02,7.59E-03,5.50E-03,3.86E-03,2.65E-03,1.78E-03,1.16E-03,7.47E-04,4.56E-04,2.79E-04,1.66E-04,9.44E-05,5.50E-05,2.99E-05,1.72E-05,8.57E-06,4.84E-06,2.10E-06,1.24E-06,5.32E-07,2.36E-07),
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



