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
process.GlobalTag.globaltag = '80X_mcRun2_asymptotic_2016_v3'

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
        #'/store/relval/CMSSW_7_6_0_pre7/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/76X_mcRun2_asymptotic_v5-v1/00000/7E692CF1-2971-E511-9609-0025905A497A.root',
        #'/store/relval/CMSSW_7_6_0_pre7/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/76X_mcRun2_asymptotic_v5-v1/00000/B4DD46D7-2971-E511-B4DD-0025905A4964.root' 
        #for MINIAODtests 
        #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/MINIAODSIM/PU50ns_PRE_LS172_V16-v1/00000/9886ACB4-F45E-E411-9E5D-02163E00F01E.root' 
        '/store/relval/CMSSW_8_0_3/RelValProdTTbar_13/MINIAODSIM/80X_mcRun2_asymptotic_2016_v3_gs7120p2NewGTv3-v1/00000/607105FE-D0EF-E511-BBC0-0CC47A78A426.root'
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
                     #process.dump*
                     #process.metPreValidSeq*
                     #process.METValidation
                     #for MiniAOD
                     process.METValidationMiniAOD
                     *process.METPostProcessor
                     *process.METPostProcessorHarvesting
                     *process.dqmSaver
)


