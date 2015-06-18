import os
import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
#from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

#process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = 'MCRUN2_74_V7::All'

#process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

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
       #test HI sequence for jets
       #'/store/relval/CMSSW_7_3_0_pre1/RelValQCD_Pt_80_120_13_HI/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/5C15CC80-0B5A-E411-AF4B-02163E00ECD2.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValQCD_Pt_80_120_13_HI/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/FC51FED6-B559-E411-9131-02163E006D72.root' 
] );

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")
process.load("Validation.RecoJets.JetPostProcessor_cff")
#process.load("Validation.RecoHI.JetValidationHeavyIons_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)
                       
process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/JETValidation'
process.dqmSaver.workflow = Workflow


process.p1 = cms.Path(
                      #--- Standard sequence
                      #process.hiJetValidation
                      #first part issued to take care of JEC
                      process.jetPreValidSeq*
                      process.JetValidation
                      #for MiniAOD
                      #process.JetValidationMiniAOD
                      *process.JetPostProcessor
                      *process.dqmSaver
)

