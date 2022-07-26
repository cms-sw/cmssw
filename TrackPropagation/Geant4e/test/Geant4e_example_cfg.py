import FWCore.ParameterSet.Config as cms

process = cms.Process("G4eRefit")


process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.default = cms.untracked.PSet(ERROR = cms.untracked.PSet(limit = cms.untracked.int32(5)))
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_12_4_0/RelValSingleMuPt10/GEN-SIM-RECO/124X_mcRun3_2022_realistic_v5-v1/2580000/7a892809-eb85-4c21-8fc2-3723b4f85c85.root'
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

process.g4RefitPath = cms.Path( process.MeasurementTrackerEvent * process.geant4eTrackRefit )
process.e = cms.EndPath(process.out)
process.schedule = cms.Schedule( process.g4RefitPath, process.e )


