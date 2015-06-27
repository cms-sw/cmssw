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
process.GlobalTag.globaltag = 'MCRUN2_75_V5'

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
       #'/store/relval/CMSSW_7_5_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/32484B3F-0E0B-E511-8191-0025905A60CE.root',
       #'/store/relval/CMSSW_7_5_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/38D6C20E-100B-E511-B21B-0025905938D4.root'
        #for MINIAODtests  
       '/store/relval/CMSSW_7_5_0_pre2/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/MCRUN2_74_V7-v1/00000/386DC497-C0E3-E411-A4FA-0025905A60F4.root' 
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
                     #process.METValidation*
                     #for MiniAOD
                     process.METValidationMiniAOD*
                     process.METPostProcessor
                     *process.METPostProcessorHarvesting
                     *process.dqmSaver
)


