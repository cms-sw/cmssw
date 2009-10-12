import FWCore.ParameterSet.Config as cms

process = cms.Process('TrackingTruthPlayback')

# Playback
process.load("SimGeneral.TrackingAnalysis.Playback_cfi")
# TrackingTruth
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

# Output definition
process.output = cms.OutputModule(
  'PoolOutputModule',
  fileName = cms.untracked.string('TrackingTruth.root'),
  outputCommands = cms.untracked.vstring(
    'keep edmHepMCProduct_source_*_*',
    'keep *_mergedtruth__*',
    'keep *_mergedtruth_*_*'
  )
)

process.path = cms.Path(process.mix*process.trackingParticles)
process.outpath = cms.EndPath(process.output)

# Input definition
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/1A0FD639-1B86-DD11-A3C0-000423D99614.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/26C81217-1E86-DD11-9B68-001617C3B654.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/2E8E22CE-1F86-DD11-BC6E-000423D98844.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/3AD747D6-1F86-DD11-AF6B-000423D9939C.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/3E71109C-1686-DD11-888F-001617E30E2C.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/40155ED4-1786-DD11-B117-000423D98C20.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/42AE0765-1F86-DD11-BB84-001617DBD5B2.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/4CD37415-1986-DD11-86A4-001617C3B6FE.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/7210024F-2186-DD11-AB5F-001617E30F48.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/806E680F-2086-DD11-B4E3-001617DBCF6A.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/8E3E9B4E-2086-DD11-B37F-000423D94494.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/90787505-1C86-DD11-B207-000423D6CAF2.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/9254F0ED-1686-DD11-A574-001617C3B5F4.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/948BE3F6-1886-DD11-BC85-000423D9880C.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/9A3F5BE4-1786-DD11-AE26-000423D98BE8.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/9E1CEB35-1986-DD11-9C95-000423D9863C.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/9E1E8199-1D86-DD11-ADD5-001617DBD332.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/B084328E-1D86-DD11-A62D-000423D94534.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/B0E48856-1C86-DD11-A5B4-00161757BF42.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/B28D6FA4-1E86-DD11-8E98-000423D952C0.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/BE2B548C-1B86-DD11-BE99-000423D99AAA.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/C2D2E863-1886-DD11-A62F-001617DF785A.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/C6317E19-1F86-DD11-B7D7-001617E30F4C.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/C883934C-1986-DD11-BBB9-000423D98868.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/F089AD31-1E86-DD11-8029-001617E30D12.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/F8989792-2086-DD11-8FB8-0019DB29C620.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/F8F91D68-1886-DD11-B0CB-001617C3B76A.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/FEEBFDA0-2086-DD11-8969-000423D952C0.root',
       '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/689151E5-0487-DD11-B613-000423D98800.root' ] );


