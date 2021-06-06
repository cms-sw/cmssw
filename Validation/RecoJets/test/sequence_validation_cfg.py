import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("METVALIDATION")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = '76X_mcRun2_asymptotic_v4'

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#
#
# DQM
#

process.load("Validation.RecoJets.JetValidation_cff")
process.load("Validation.RecoJets.JetPostProcessor_cff")


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_6_0_pre7/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/76X_mcRun2_asymptotic_v5-v1/00000/7E692CF1-2971-E511-9609-0025905A497A.root',
       '/store/relval/CMSSW_7_6_0_pre7/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/76X_mcRun2_asymptotic_v5-v1/00000/B4DD46D7-2971-E511-B4DD-0025905A4964.root' 
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/5C506D9D-AA6B-E511-8AB0-003048FFD732.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/5E68FFC6-B66B-E511-9085-0025905A6084.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/724E805D-816C-E511-A163-0025905B8562.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/88117A83-A76B-E511-9A30-002590596490.root'
        #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/0847033F-706C-E511-8170-0025905A60CE.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/98244E0E-C56C-E511-A5AF-0025905B85EE.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/B6EB8FF6-C46C-E511-911A-0025905B8592.root',
       #'/store/relval/CMSSW_7_6_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v4-v1/00000/FABE0A3D-6F6C-E511-A404-0025905A6076.root' 
] );

process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/JetValidation'
process.dqmSaver.workflow = Workflow
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(#process.dump*
                     #for RECO
                     #process.jetPreValidSeq*
                     #process.JetValidation*
                     #for MiniAOD
                     process.JetValidationMiniAOD*
                     process.JetPostProcessor
                     *process.dqmSaver
)


