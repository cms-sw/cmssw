import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("G4eRefit",Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.default = cms.untracked.PSet(ERROR = cms.untracked.PSet(limit = cms.untracked.int32(5)))
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_14_0_0_pre2/RelValSingleMuPt10/GEN-SIM-RECO/133X_mcRun3_2023_realistic_v3_STD-v2/2590000/da5cf255-2d65-41ec-ac89-45daf33c66d3.root',
    ),
)

process.load("TrackPropagation.Geant4e.geantRefit_cff")
process.Geant4eTrackRefitter.src = cms.InputTag("generalTracks")
process.Geant4eTrackRefitter.usePropagatorForPCA = cms.bool(True)
process.g4RefitPath = cms.Path( process.MeasurementTrackerEvent * process.geant4eTrackRefit )

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
  'keep *'
  ),
  fileName = cms.untracked.string( 'test.root' )
)

process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule( process.g4RefitPath, process.e )


