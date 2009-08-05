# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("RecoBTag.Configuration.RecoBTag_cff")


process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  

process.load("Validation.RecoB.bTagAnalysis_cfi")
process.bTagValidation.jetMCSrc = 'IC5byValAlgo'
process.bTagValidation.allHistograms = True 
#process.bTagValidation.fastMC = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.plots = cms.Path(process.myPartons* process.iterativeCone5Flavour * process.bTagValidation*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/CAAA36CC-9841-DE11-A587-0019B9F730D2.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/B47CEC98-E641-DE11-9999-001D09F2437B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/98E6DFEA-9941-DE11-B198-001D09F25438.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/74475B04-9B41-DE11-A6CB-001D09F24D8A.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/6A9F37C0-9B41-DE11-8334-001D09F28C1E.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/4CF9716F-9E41-DE11-A0BA-001D09F25438.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/18B38D57-9C41-DE11-9DD9-001D09F250AF.root' ]

