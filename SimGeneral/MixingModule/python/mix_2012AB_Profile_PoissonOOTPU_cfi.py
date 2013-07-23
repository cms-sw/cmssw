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
                  5.99688E-12,
                  3.9153E-10,
                  3.71925E-07,
                  3.03413E-05,
                  7.34144E-05,
                  0.000347794,
                  0.001975866,
                  0.00517831,
                  0.010558132,
                  0.018312126,
                  0.028852472,
                  0.040969532,
                  0.050957346,
                  0.058116316,
                  0.063832662,
                  0.067924013,
                  0.068664032,
                  0.06634696,
                  0.06258012,
                  0.058673398,
                  0.055389375,
                  0.05287113,
                  0.050854502,
                  0.048745749,
                  0.045577918,
                  0.040642954,
                  0.033995191,
                  0.026305836,
                  0.018543156,
                  0.011719526,
                  0.006558513,
                  0.003227318,
                  0.00139409,
                  0.000529933,
                  0.000178348,
                  5.36943E-05,
                  1.46867E-05,
                  3.72253E-06,
                  8.91697E-07,
                  2.04492E-07,
                  4.49872E-08,
                  9.4322E-09,
                  1.86873E-09,
                  3.48142E-10,
                  6.10474E-11,
                  1.01401E-11,
                  1.61054E-12,
                  2.4678E-13,
                  3.66567E-14,
                  5.27305E-15,
                  7.30257E-16,
                  9.65776E-17,
                  1.20959E-17,
                  1.41538E-18,
                  1.59156E-19,
                  1.00245E-20,
                  0,
                  0,
                  0,
                  0
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



