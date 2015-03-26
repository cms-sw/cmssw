
import sys
import os

import electronDbsDiscovery

import FWCore.ParameterSet.Config as cms

from TrackingTools.Configuration.TrackingTools_cff import *

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring(),
)

process.source.fileNames.extend(electronDbsDiscovery.search())
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

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

process.p = cms.Path(process.gsfElectrons)
process.outpath = cms.EndPath(process.out)
process.out.fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'])
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']+'::All'



