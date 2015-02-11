import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("METVALIDATION")

process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = 'PRE_LS172_V16::All'

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
#
#
# DQM
#

process.load("Validation.RecoMET.METRelValForDQM_cff")


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       #for RECO
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/22A79853-D85E-E411-BAA9-02163E00C055.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/28923A16-C95E-E411-871C-02163E00FFCE.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/307F76E6-E05E-E411-90AF-02163E00B036.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/4E03E1A5-CE5E-E411-AE0F-02163E008BE3.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/689DCC5B-D35E-E411-A720-02163E00D13A.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/CC3F6060-DA5E-E411-BA7C-02163E0105B8.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/D470466A-C55E-E411-A382-02163E00EB5D.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/E2A34427-E75E-E411-ABBA-02163E008DD3.root',
        '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/FCE96BE5-F15E-E411-BD38-02163E00D13A.root'
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
                     *process.dqmSaver
)


