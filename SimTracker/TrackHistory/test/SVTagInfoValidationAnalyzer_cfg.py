import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackOriginAnalyzerTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.SecondaryVertexTagInfoProxy_cff")
process.load("SimTracker.TrackHistory.VertexClassifier_cff")

process.add_(
  cms.Service("TFileService",
      fileName = cms.string("SVTagInfoValidation.root")
  )
)

process.svTagInfoValidationAnalyzer = cms.EDFilter("SVTagInfoValidationAnalyzer",
    process.vertexClassifier,
    svTagInfoProducer = cms.untracked.InputTag('secondaryVertexTagInfos')
)

process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.svTagInfoValidationAnalyzer.enableSimToReco = cms.untracked.bool(True)
process.svTagInfoValidationAnalyzer.vertexProducer = cms.untracked.InputTag('svTagInfoProxy')

process.p = cms.Path(process.playback * process.svTagInfoProxy * process.svTagInfoValidationAnalyzer)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/BCE77A07-AC16-DE11-80B9-000423D986A8.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/B0D94AFE-3616-DE11-BFD5-000423D9880C.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/76E8D7B2-5216-DE11-8A7A-000423D174FE.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/6ED9476F-4C16-DE11-8BFC-001617C3B76A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/289FC85A-4216-DE11-ACEE-000423D98844.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/12C01897-4616-DE11-8AA7-000423D98B5C.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/00E48100-3A16-DE11-A693-001617DBCF6A.root' ] );


secFiles.extend( [
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/FCBB0F1B-3616-DE11-8335-0016177CA778.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/F8D0F7AB-6A16-DE11-A4A1-001617C3B76E.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/ECAD3734-4F16-DE11-93EE-00161757BF42.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/E8EBBF47-3816-DE11-BD8F-000423D98800.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/E8315265-3316-DE11-B8E8-000423D6C8EE.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/E4540FBC-3C16-DE11-B32E-001617E30D0A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/DE76A682-4516-DE11-955D-001617DBD224.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/C4851304-4916-DE11-8FA4-001617C3B65A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/AC245423-4916-DE11-BBA8-000423D991F0.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/A2DD0DEA-4116-DE11-BB1C-001617DBCF6A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/98A318F4-3416-DE11-8305-000423D94AA8.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/74CF9B73-3916-DE11-9EEF-000423D985E4.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/689349B5-4616-DE11-81F4-000423D991F0.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/627B424D-4216-DE11-B135-001617C3B79A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/580F53DC-4F16-DE11-8A58-000423D94534.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/569AE526-5316-DE11-9596-000423D944F0.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/52C2A955-3716-DE11-87D2-000423D99A8E.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/44601F6F-4A16-DE11-B830-001617E30D00.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/4250F67F-4C16-DE11-95D4-000423D98DC4.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/3AA6EEA4-3B16-DE11-B35F-001617C3B654.root'] );
