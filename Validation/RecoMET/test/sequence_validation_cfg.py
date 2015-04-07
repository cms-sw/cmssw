import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("METVALIDATION")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = 'MCRUN2_74_V7::All'

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
       '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/48E3FDFE-3DBD-E411-9B99-0025905A613C.root',
       '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/706A960F-54BD-E411-8561-00261894384F.root',
       '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/E4EF6410-54BD-E411-8838-002590593920.root',
       '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/FA18AB00-3EBD-E411-AAE8-0025905A608A.root'
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


