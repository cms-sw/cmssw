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
                      2.30103E-06,
                      6.10952E-06,
                      1.20637E-05,
                      2.55143E-05,
                      3.22897E-05,
                      0.000120076,
                      0.000800727,
                      0.003176761,
                      0.008762044,
                      0.018318462,
                      0.032467397,
                      0.050617606,
                      0.064967885,
                      0.072194727,
                      0.076272353,
                      0.079908524,
                      0.08275316,
                      0.083946479,
                      0.083972048,
                      0.082763689,
                      0.080056866,
                      0.076035269,
                      0.071231913,
                      0.065785365,
                      0.059393488,
                      0.052004578,
                      0.044060277,
                      0.036161288,
                      0.028712778,
                      0.021943183,
                      0.016047885,
                      0.011184911,
                      0.007401996,
                      0.004628635,
                      0.002717483,
                      0.001488303,
                      0.000757141,
                      0.000357766,
                      0.000157852,
                      6.5729E-05,
                      2.62131E-05,
                      1.01753E-05,
                      3.89905E-06,
                      1.48845E-06,
                      5.67959E-07,
                      2.16348E-07,
                      8.19791E-08,
                      3.07892E-08,
                      1.1436E-08,
                      4.19872E-09,
                      1.52513E-09,
                      5.48804E-10,
                      1.95797E-10,
                      6.92424E-11,
                      2.42383E-11,
                      8.37978E-12,
                      2.85421E-12,
                      9.55583E-13,
                      3.13918E-13,
                      1.01066E-13
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



