import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for tqaflayer1 & 2 production from
# fullsim for semi-leptonic ttbar events 
#-------------------------------------------------
process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.categories = cms.untracked.vstring('TEST')

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #PAT test sample
    #'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_1_X_2008-07-08_STARTUP_V4-AODSIM.100.root'
    #210 RelVal sample
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/061DC5C9-8962-DD11-AB87-001617C3B5F4.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/1846FB92-8B62-DD11-BF46-001617C3B5D8.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/28BA9967-8A62-DD11-8CBC-001617C3B6CC.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/3CE74890-8A62-DD11-A309-001617C3B79A.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/6CE93E47-E262-DD11-99D9-000423D6BA18.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/8404EE20-8B62-DD11-A6AD-001617C3B6C6.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/A2111BED-8E62-DD11-9AB8-000423D98804.root',
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RECO/STARTUP_V4_v3/0001/DEFB6B46-8C62-DD11-8643-001617C3B79A.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')

# Magnetic field now needs to be in the high-level py
process.load("Configuration.StandardSequences.MagneticField_cff")


#-------------------------------------------------
# tqaf configuration; if the TQAF Layer 1 is
# already in place yuo can comment the following
# two lines
#-------------------------------------------------

## std sequence for tqaf layer1
process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_full_cff")
process.p0 = cms.Path(process.tqafLayer1)

#-------------------------------------------------
# analyze electrons
#-------------------------------------------------
from TopQuarkAnalysis.Examples.TopElecAnalyzer_cfi import analyzeElec
process.analyzeElec = analyzeElec

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopElec.root')
)

## end path   
process.p1 = cms.Path(process.analyzeElec)

