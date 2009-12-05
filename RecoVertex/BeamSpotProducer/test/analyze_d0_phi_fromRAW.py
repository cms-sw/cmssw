import FWCore.ParameterSet.Config as cms
process = cms.Process("d0phi")

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoTracker.Configuration.RecoTrackerNotStandard_cff")
process.MeasurementTracker.pixelClusterProducer = cms.string("")

process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = 'ctfPixelLess'

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.GlobalTag.globaltag = 'MC_31X_V9::All'

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")
process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_pixelLess_cff")
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/00864FB5-E1BC-DE11-8613-0018F3D0961A.root',
    '/store/relval/CMSSW_3_3_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/0012B75C-DDBC-DE11-81E2-002618943921.root'

    ] );

secFiles.extend( [
    ] )

process.source = source

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

# Path and EndPath definitions
process.pixelLessTrack_step = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.ctfTracksPixelLess)
process.tracking_step = cms.Sequence(process.siPixelDigis*process.siStripDigis*process.trackerlocalreco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
                               outputCommands = cms.untracked.vstring(
                                'drop *',
                                'keep *_offlineBeamSpot_*_*',
                                'keep *_offlinePrimaryVertices_*_*',
                                'keep *_ctfPixelLess_*_*'
                               )
                              )
process.end = cms.EndPath(process.out)

process.schedule = cms.Path(process.pixelLessTrack_step*process.offlinePrimaryVertices*process.d0_phi_analyzer_pixelless)
#process.schedule = cms.Path(process.pixelLessTrack_step*process.offlinePrimaryVertices*process.d0_phi_analyzer_pixelless*process.end)
#process.schedule = cms.Path(process.tracking_step*process.offlinePrimaryVertices*process.d0_phi_analyzer)
