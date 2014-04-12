from ElectronRedoFromRaw_driver_cfg import *

import os, sys
import DQMOffline.EGamma.electronDataDiscovery as dd

process.source.fileNames = cms.untracked.vstring()
process.source.fileNames.extend(dd.search())
process.source.secondaryFileNames = cms.untracked.vstring()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

uncleanedOnlyElectronSeeds = process.ecalDrivenElectronSeeds.clone()
uncleanedOnlyElectronSeeds.barrelSuperClusters = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridSuperClusters")

process.RECOSIMoutput.fileName = cms.untracked.string(os.environ['TEST_HISTOS_FILE'])
process.RECOSIMoutput.outputCommands = cms.untracked.vstring('drop *', 
  'keep recoBeamSpot*_*_*_*',
  'keep recoGenParticle*_*_*_*',
  'keep *HepMCProduct_*_*_*',
  'keep recoElectronSeed*_*_*_*', 
  'keep recoSuperCluster*_*_*_*', 
  'keep recoTrack*_*_*_*', 
  'keep recoGsfTrack*_*_*_*', 
  'keep *_iterativeCone5GenJets_*_*', 
  'keep *_iterativeCone5CaloJets_*_*', 
  'keep *_gsfElectronCores_*_*', 
  'keep *_gsfElectrons_*_*'
)

#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond[os.environ['TEST_GLOBAL_AUTOCOND']]

process.dumpPython(None)

