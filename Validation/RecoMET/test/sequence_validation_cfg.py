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
process.GlobalTag.globaltag = '75X_mcRun2_asymptotic_v1'

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
       #for RECO
       '/store/relval/CMSSW_7_5_0_pre6/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/06121D17-661B-E511-AE2A-0025905A60B2.root',
       '/store/relval/CMSSW_7_5_0_pre6/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/103A92CC-681B-E511-A923-002618943935.root',
       '/store/relval/CMSSW_7_5_0_pre6/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/16752615-911B-E511-991B-0025905A613C.root',
       '/store/relval/CMSSW_7_5_0_pre6/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/75X_mcRun2_asymptotic_v1-v1/00000/1C073F39-671B-E511-94BE-0025905A60D0.root'
        #for MINIAODtests 
        #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/MINIAODSIM/PU50ns_PRE_LS172_V16-v1/00000/9886ACB4-F45E-E411-9E5D-02163E00F01E.root' 
] );

process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/METValidation'
process.dqmSaver.workflow = Workflow
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


process.p = cms.Path(#for RECO
                     process.metPreValidSeq*
                     process.METValidation
                     #for MiniAOD
                     #process.METValidationMiniAOD
                     process.METPostProcessor
                     *process.METPostProcessorHarvesting
                     *process.dqmSaver
)


