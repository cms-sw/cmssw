
#
# TO BE DONE : ADD WHAT IS LACKING FOR ISO FROM DEPOSITS
#

import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dbs
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronRedoFromTrackSeeds")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond[os.environ['TEST_GLOBAL_AUTOCOND']]

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dbs.search())

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoBeamSpot*_*_*_*',
        'keep recoGenParticle*_*_*_*',
        'keep *HepMCProduct_*_*_*',
        'keep recoElectronSeed*_*_*_*', 
        'keep recoSuperCluster*_*_*_*', 
        'keep recoTrack*_*_*_*', 
        'keep recoGsfTrack*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_gsfElectronCores_*_*', 
        'keep *_gsfElectrons_*_*'
    ),
    fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'])
)

process.newCombinedSeeds.seedCollections = cms.VInputTag(
     cms.InputTag('initialStepSeeds'),
     cms.InputTag('pixelPairStepSeeds'),
     cms.InputTag('mixedTripletStepSeeds'),
     cms.InputTag('pixelLessStepSeeds')
)

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.iterTracking*process.trackCollectionMerging*process.newCombinedSeeds*process.electronSeeds*process.electronCkfTrackCandidates*process.electronGsfTracks*process.particleFlowCluster*process.particleFlowTrackWithDisplacedVertex*process.gsfEcalDrivenElectronSequence*process.muonrecoComplete*process.particleFlowReco*process.gsfElectronMergingSequence)

process.outpath = cms.EndPath(process.out)



