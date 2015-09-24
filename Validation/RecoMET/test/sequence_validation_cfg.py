import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("METVALIDATION")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = '74X_mcRun2_asymptotic_v2'

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#
#
# DQM
#

process.load("Validation.RecoMET.METRelValForDQM_cff")
process.load("Validation.RecoMET.METPostProcessor_cff")


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/006F3660-4B5E-E511-B8FD-0025905B8596.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/1E34D76A-7B5E-E511-AD5E-0025905A6138.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/22EF90FD-4C5E-E511-952A-0025905B8590.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/62258A2B-7D5E-E511-985C-0025905A6094.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/6A040A44-495E-E511-82C4-0025905A605E.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/70101C9E-4D5E-E511-AC66-0025905B855C.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/7467606A-4B5E-E511-8606-00261894390E.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/AE731747-495E-E511-AA22-0025905B8576.root',
       '/store/relval/CMSSW_7_4_12/RelValTTbar_13/GEN-SIM-RECO/PU25ns_74X_mcRun2_asymptotic_v2_v2-v1/00000/D0F8E44D-505E-E511-BFF0-002618943809.root'
] );

process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/METValidation'
process.dqmSaver.workflow = Workflow
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.p = cms.Path(#for RECO
                     process.metPreValidSeq*
                     process.METValidation*
                     #for MiniAOD
                     #process.METValidationMiniAOD*
                     process.METPostProcessor
                     *process.METPostProcessorHarvesting
                     *process.dqmSaver
)


