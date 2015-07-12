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
process.GlobalTag.globaltag = 'MCRUN2_74_V9'

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
        #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/28F53E5E-321D-E511-AEF1-0026189438F7.root',
       #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/4236E25F-321D-E511-92B6-0026189438B0.root'
       #for RECO
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/403FA79E-251A-E511-B21A-0025905B855C.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/623B1740-551A-E511-8A61-0025905A60CA.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/6AEC1D6D-361A-E511-8AFF-0025905938A8.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/801ABCFB-5B1A-E511-BB68-0025905A4964.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/9429572C-221A-E511-8613-0025905A6064.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/B6B7C7EA-1E1B-E511-8472-0025905B8576.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/D6833B78-3C1A-E511-9D9B-0026189438B5.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/DA508AF0-4E1A-E511-A655-0025905A60E0.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/F479010F-491A-E511-98F9-0025905938B4.root',
       '/store/relval/CMSSW_7_4_6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/F837DA4C-561B-E511-BB00-0025905A6138.root' 
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


