# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("RecoBTag.Configuration.RecoBTag_cff")

process.load("Validation.RecoB.bTagAnalysis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(300)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.plots = cms.Path(process.bTagValidation*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'RelVal'
process.PoolSource.fileNames = ['/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/08765709-5826-DD11-9CE8-000423D94700.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/0E9C16AC-5626-DD11-954E-001617DC1F70.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/1C4313FF-5626-DD11-A4E2-001617E30D0A.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/2AFF9647-5726-DD11-A448-000423D986A8.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/2C7FB344-5726-DD11-9305-000423D6CA72.root', 
    '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/501A1388-5926-DD11-8D7B-001617E30F4C.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/648B3A05-5826-DD11-BB66-000423D6B42C.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/807153D6-5726-DD11-8FB7-001617E30F50.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/8869FBD3-5726-DD11-8F28-000423D6CAF2.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/96EFF712-5B26-DD11-9016-001617E30E28.root', 
    '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/A0F81248-5726-DD11-BD40-000423D6B42C.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/AAD735D2-5726-DD11-ACBF-001617E30E2C.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/B22DBB07-5826-DD11-8A8C-000423D992A4.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/BC3B21B8-5626-DD11-9FE3-000423D6B5C4.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/CE14BFB1-5626-DD11-9AAC-001617DBD5AC.root', 
    '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/E6266C77-5A26-DD11-9E89-000423D6B42C.root', '/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/F49B15D5-5726-DD11-B0D1-001617DBD230.root']


