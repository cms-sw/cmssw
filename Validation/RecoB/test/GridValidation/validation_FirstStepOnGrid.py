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
process.load("Validation.RecoB.bTagAnalysis_firststep_cfi")
process.bTagValidationFirstStep.jetMCSrc = 'IC5byValAlgo'
process.bTagValidationFirstStep.allHistograms = True 
#process.bTagValidation.fastMC = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.EDM = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      "keep *_MEtoEDMConverter_*_*"),
                               fileName = cms.untracked.string('MEtoEDMConverter.root')
                               )
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.plots = cms.Path(process.myPartons* process.iterativeCone5Flavour * process.bTagValidationFirstStep* process.MEtoEDMConverter)

process.outpath = cms.EndPath(process.EDM)


process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'RelVal'
process.PoolSource.fileNames = [
#       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/CAAA36CC-9841-DE11-A587-0019B9F730D2.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/B47CEC98-E641-DE11-9999-001D09F2437B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/98E6DFEA-9941-DE11-B198-001D09F25438.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/74475B04-9B41-DE11-A6CB-001D09F24D8A.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/6A9F37C0-9B41-DE11-8334-001D09F28C1E.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/4CF9716F-9E41-DE11-A0BA-001D09F25438.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0004/18B38D57-9C41-DE11-9DD9-001D09F250AF.root' ]

