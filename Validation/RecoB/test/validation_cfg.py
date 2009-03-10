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


process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  
process.bTagValidation.jetMCSrc = 'IC5byValAlgo'
process.bTagValidation.allHistograms = True 
process.bTagValidation.fastMC = True


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
process.dqmSaver.convention = 'RelVal'
process.PoolSource.fileNames = [
           '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/00E9B0FB-98C4-DD11-AF51-0030487A322E.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/28E6A7DA-98C4-DD11-AE6E-001D09F2512C.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/3802DDF4-9FC4-DD11-A9AC-001D09F24600.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/5CA764F6-98C4-DD11-A05F-001D09F241B4.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0002/A6D75ADB-98C4-DD11-B05E-0019B9F704D6.root',
        '/store/relval/CMSSW_2_2_1/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v2/0003/72C99195-FDC4-DD11-8F03-001617C3B79A.root'
           ]

