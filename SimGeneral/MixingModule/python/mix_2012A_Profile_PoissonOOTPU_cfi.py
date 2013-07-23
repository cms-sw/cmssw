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
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59),
          probValue = cms.vdouble(
                  2.90E-15,
                  2.00E-09,
                  1.57E-06,
                  1.46E-04,
                  3.10E-04,
                  4.83E-05,
                  5.74E-05,
                  5.36E-05,
                  4.45E-04,
                  4.33E-03,
                  2.00E-02,
                  4.26E-02,
                  6.00E-02,
                  7.35E-02,
                  8.63E-02,
                  9.54E-02,
                  9.82E-02,
                  9.41E-02,
                  8.57E-02,
                  7.62E-02,
                  6.66E-02,
                  5.64E-02,
                  4.57E-02,
                  3.51E-02,
                  2.49E-02,
                  1.59E-02,
                  9.08E-03,
                  4.74E-03,
                  2.30E-03,
                  1.07E-03,
                  4.78E-04,
                  2.06E-04,
                  8.52E-05,
                  3.32E-05,
                  1.20E-05,
                  3.99E-06,
                  1.21E-06,
                  3.29E-07,
                  8.05E-08,
                  1.76E-08,
                  3.45E-09,
                  5.99E-10,
                  9.26E-11,
                  1.27E-11,
                  1.54E-12,
                  1.66E-13,
                  1.58E-14,
                  1.33E-15,
                  9.91E-17,
                  6.53E-18,
                  3.84E-19,
                  1.65E-20,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00,
                  0.00E+00
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



