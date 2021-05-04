import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *

process = cms.Process("PRODVAL1")
process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(56789)
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
   	'/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/3AA6EEA4-3B16-DE11-B35F-001617C3B654.root')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    
    maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
    ),
    
    input = cms.SecSource("EmbeddedRootSource",
    fileNames = cms.untracked.vstring(
   	'/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/3AA6EEA4-3B16-DE11-B35F-001617C3B654.root'),

        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),

        maxEventsToSkip = cms.untracked.uint32(0),
        
    ),
    
    maxBunch = cms.int32(12345),
    minBunch = cms.int32(12345),
    bunchspace = cms.int32(25),
    checktof = cms.bool(False),
    Label = cms.string(''),
    
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

process.test = cms.EDAnalyzer("TestSuite",
    maxBunch = cms.int32(34567),
    BunchNr = cms.int32(12345),
    minBunch = cms.int32(23456),
    fileName = cms.string('histos.root')
)

process.p = cms.Path(process.mix+process.test)
#process.outpath = cms.EndPath(process.out)


