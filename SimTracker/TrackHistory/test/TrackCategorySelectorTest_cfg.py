import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackCategorySelectorTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.TrackClassifier_cff")

from SimTracker.TrackHistory.CategorySelectors_cff import * 

process.trackSelector = TrackCategorySelector( 
    src = cms.InputTag('generalTracks'),
    cut = cms.string("is('BWeakDecay')")
)

process.trackHistoryAnalyzer = cms.EDAnalyzer("TrackHistoryAnalyzer",
    process.trackClassifier
)

process.trackHistoryAnalyzer.trackProducer = 'trackSelector'

process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.p = cms.Path(process.playback * process.trackSelector * process.trackHistoryAnalyzer)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( ( 
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0005/2A400E33-B0E2-DD11-94F7-000423D60FF6.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0005/327A7D26-B1E2-DD11-8A2F-001617C3B77C.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0005/4297C3C7-B0E2-DD11-8585-001617DBD556.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0005/528BE768-B0E2-DD11-95BF-000423D98B28.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0006/54A51252-B7E2-DD11-A9D8-000423D9870C.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0006/BE406FCD-B1E2-DD11-8BDF-001617DBD332.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0006/F66715A5-D9E2-DD11-9313-000423D94908.root' 
) );

secFiles.extend( (
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/22F84CC1-B0E2-DD11-97FA-001617E30D00.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/447729C6-B0E2-DD11-A066-001617C3B6C6.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/76FF1F2F-B0E2-DD11-A000-000423D98834.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/AA3C812F-B0E2-DD11-A975-001617DBD332.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/AAC8158C-B0E2-DD11-9D76-001617E30D40.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/ACDF60CE-AFE2-DD11-90F7-000423D9A2AE.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/C451B88C-B0E2-DD11-AD23-000423D991F0.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/D8AEB2BC-B0E2-DD11-9D6D-001617C3B5F4.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/EC4E792E-B0E2-DD11-BB5A-001617E30D00.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/ECA45CBD-B0E2-DD11-9AFA-001617DBD230.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/F41A298C-B0E2-DD11-86DE-001617E30D52.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/0AB5DE51-B7E2-DD11-8C0B-000423D6BA18.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/100FDD4E-B1E2-DD11-B52D-001617E30D52.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/263963A1-D9E2-DD11-A704-001617DBD288.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/26CB5F22-B1E2-DD11-872E-001617C3B66C.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/54B25E22-B1E2-DD11-8503-001617E30F48.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/A2B2FDCA-B1E2-DD11-A0C7-001617E30D06.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/A6BDE6E4-B3E2-DD11-AB5B-000423D992DC.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/DA5D2953-B1E2-DD11-95DD-001617DBD316.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/EA30EB57-B2E2-DD11-9928-001617C3B778.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0006/F6C31724-B1E2-DD11-A264-001617DBD5AC.root' 
) )

