import FWCore.ParameterSet.Config as cms
process = cms.Process("d0phi")

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
#process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('RecoTracker/Configuration/RecoTrackerNotStandard_cff')

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.GlobalTag.globaltag = 'MC_31X_V3::All'

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")
process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_pixelLess_cff")
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/EC49FE0B-DB90-DE11-8434-001D09F2545B.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v2/0011/C0F33D87-D990-DE11-8490-000423D6B444.root'
    ] );

secFiles.extend( [
    ] )

process.source = source

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Path and EndPath definitions
process.raw2digi_step = cms.Sequence(process.siPixelDigis*
                                     process.SiStripRawToDigis
                                    )

process.pretracking_step = cms.Sequence(process.trackerlocalreco+
                                        process.offlineBeamSpot+
                                        process.recopixelvertexing
                                       )

process.pixelLessTrack_step = cms.Path(process.raw2digi_step+process.pretracking_step*process.ctfTracksPixelLess)
process.tracking_step = cms.Path(process.raw2digi_step+process.pretracking_step*process.ckftracks)
process.p = cms.Path(process.d0_phi_analyzer)
process.p1 = cms.Path(process.d0_phi_analyzer_pixelless)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root'),
                               outputCommands = cms.untracked.vstring(
                                'drop *',
                                'keep *_offlineBeamSpot_*_*',
                                'keep *_ctfPixelLess_*_*'
                               )
                              )
process.end = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.pixelLessTrack_step,process.p1)
#process.schedule = cms.Schedule(process.pixelLessTrack_step,process.p1,process.end)
#process.schedule = cms.Schedule(process.tracking_step,process.p)
