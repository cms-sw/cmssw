from ElectronRedoFromRaw_driver_cfg import *

import os
import electronDbsDiscovery

process.source.fileNames = cms.untracked.vstring()
process.source.fileNames.extend(electronDbsDiscovery.search())
process.source.secondaryFileNames = cms.untracked.vstring()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.RECOSIMoutput.fileName = cms.untracked.string(os.environ['TEST_OUTPUT_FILE'])
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
#  'keep *_ecalDrivenGsfElectronCores_*_*', 
  'keep *_gsfElectronCores_*_*', 
  'keep *_gsfElectrons_*_*'
)

#from Configuration.PyReleaseValidation.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['mc']
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']+'::All'

#
#process.out = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *', 
#        'keep recoBeamSpot*_*_*_*',
#        'keep recoGenParticle*_*_*_*',
#        'keep *HepMCProduct_*_*_*',
#        'keep recoElectronSeed*_*_*_*', 
#        'keep recoSuperCluster*_*_*_*', 
#        'keep recoTrack*_*_*_*', 
#        'keep recoGsfTrack*_*_*_*', 
#        'keep *_iterativeCone5GenJets_*_*', 
#        'keep *_iterativeCone5CaloJets_*_*', 
#        'keep *_gsfElectronCores_*_*', 
#        'keep *_gsfElectrons_*_*'
#    ),
#    fileName = cms.untracked.string(os.environ['TEST_OUTPUT_FILE'])
#)
#
#process.gsfElectrons.ctfTracksCheck = cms.bool(False)
#

