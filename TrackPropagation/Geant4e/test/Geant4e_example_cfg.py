import FWCore.ParameterSet.Config 

process = cms.Process("G4eRefit")


process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")


from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.default = cms.untracked.PSet(ERROR = cms.untracked.PSet(limit = cms.untracked.int32(5)))
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_4_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/4C6BF9BF-95A9-E411-ADA9-0025905A60FE.root',
       '/store/relval/CMSSW_7_4_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/DE6B164D-9FA9-E411-9BAF-0025905B85D0.root' 
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


