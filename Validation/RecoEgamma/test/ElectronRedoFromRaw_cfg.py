from ElectronRedoFromRaw_driver_cfg import *

import os, sys
import DQMOffline.EGamma.electronDbsDiscovery as dbs

process.source.fileNames = cms.untracked.vstring()
process.source.fileNames.extend(dbs.search())
process.source.secondaryFileNames = cms.untracked.vstring()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

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
  'keep *_gsfElectrons_*_*',
  'keep *_uncleanedOnly*_*_*', 
)

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

#process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']+'::All'

#process.pfElectronTranslator.MVACutBlock.MVACut = cms.double(-0.4)
#process.gsfElectrons.minMVA = cms.double(-0.4)
#process.gsfElectrons.minMVAPflow = cms.double(-0.4)

#process.ecalDrivenGsfElectronCores.useGsfPfRecTracks = cms.bool(False)
#process.gsfElectronCores.useGsfPfRecTracks = cms.bool(False)
#process.ecalDrivenGsfElectrons.useGsfPfRecTracks = cms.bool(False)
#process.gsfElectrons.useGsfPfRecTracks = cms.bool(False)

#process.particleFlow.useCalibrationsFromDB = cms.bool(False)

#process.gsfElectrons.ctfTracksCheck = cms.bool(False)


