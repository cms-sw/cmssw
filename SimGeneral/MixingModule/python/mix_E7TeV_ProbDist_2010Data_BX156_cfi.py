import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 nsec

    bunchspace = cms.int32(450), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
        seed = cms.int32(1234567),
        type = cms.string('probFunction'),
        nbPileupEvents = cms.PSet(
          probFunctionVariable = cms.vint32(0,1,2,3,4,5,6,7,8,9,10),
          probValue = cms.vdouble(0.145168,0.251419,0.251596,0.17943,0.10,0.05,0.02,0.01,0.005,0.002,0.001),
          histoFileName = cms.untracked.string('histProbFunction.root'),
          seed = cms.untracked.int32(54321)
        ),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_8_5/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0040/C4C6B18F-B6D2-DF11-80A7-002618943870.root',
                '/store/relval/CMSSW_3_8_5/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0039/6AFB33BC-E7D1-DF11-84D3-001A928116F8.root')
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



