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
                         3.58E-11,
                         3.40E-10,
                         6.54E-07,
                         3.55E-05,
                         1.28E-04,
                         2.03E-03,
                         1.17E-02,
                         3.09E-02,
                         6.26E-02,
                         1.05E-01,
                         1.52E-01,
                         2.02E-01,
                         2.44E-01,
                         2.74E-01,
                         2.95E-01,
                         3.10E-01,
                         3.12E-01,
                         3.02E-01,
                         2.88E-01,
                         2.74E-01,
                         2.64E-01,
                         2.59E-01,
                         2.58E-01,
                         2.56E-01,
                         2.47E-01,
                         2.27E-01,
                         1.94E-01,
                         1.52E-01,
                         1.08E-01,
                         6.89E-02,
                         3.87E-02,
                         1.91E-02,
                         8.24E-03,
                         3.13E-03,
                         1.05E-03,
                         3.17E-04,
                         8.65E-05,
                         2.19E-05,
                         5.24E-06,
                         1.20E-06,
                         2.65E-07,
                         5.57E-08,
                         1.11E-08,
                         2.07E-09,
                         3.63E-10,
                         6.04E-11,
                         9.60E-12,
                         1.47E-12,
                         2.19E-13,
                         3.15E-14,
                         4.36E-15,
                         5.77E-16,
                         7.22E-17,
                         8.45E-18,
                         9.50E-19,
                         5.99E-20,
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



