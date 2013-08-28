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
    #test file for Desy
    #'file:/afs/desy.de/user/r/rwolf/cms13/samples/16D75F0F-1186-DD11-80B9-000423D98C20.root'
    #219 RelVal sample at cern
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/16D75F0F-1186-DD11-80B9-000423D98C20.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/24CD41BB-1A86-DD11-9CDA-000423D98EC8.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/264D79AA-1786-DD11-9F3C-001617C3B6DC.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/327DE1B9-0E86-DD11-B7B1-000423D6C8E6.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/345AE083-1186-DD11-8D43-000423D99658.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4A0ADB7D-1086-DD11-BD16-000423D98E6C.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4E31E969-1886-DD11-8398-000423D9989E.root',
    #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4EEEA6AE-0886-DD11-90F9-000423D94990.root'
    #219 RelVal sample at desy
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/16D75F0F-1186-DD11-80B9-000423D98C20.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/24CD41BB-1A86-DD11-9CDA-000423D98EC8.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/264D79AA-1786-DD11-9F3C-001617C3B6DC.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/327DE1B9-0E86-DD11-B7B1-000423D6C8E6.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/345AE083-1186-DD11-8D43-000423D99658.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4A0ADB7D-1086-DD11-BD16-000423D98E6C.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4E31E969-1886-DD11-8398-000423D9989E.root',
    'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4EEEA6AE-0886-DD11-90F9-000423D94990.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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

## Magnetic field now needs to be in the high-level py
process.load("Configuration.StandardSequences.MagneticField_cff")


#-------------------------------------------------
# make full set of global fit  calibrated jets
#-------------------------------------------------

## make gf calibrated icone5 jets
process.load("JetMETCorrections.Configuration.GFCorrections_StepEfracParameterization_cff")
process.p1 = cms.Path(process.gfCorJetIcone5)

#-------------------------------------------------
# make full set of jetmet fact calibrated jets
#-------------------------------------------------

## make L2L3 calibrated jets
process.load("JetMETCorrections.Configuration.L2L3Corrections_iCSA08_S156_cff")
process.p2 = cms.Path(process.L2L3CorJetIcone5)

#-------------------------------------------------
# make validation
#-------------------------------------------------

## register TFile service
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('genJetClosure.root')
)

## make validation
process.load("Validation.RecoJets.sequences.GenJetClosure_cff")
process.p3 = cms.Path(process.makeAllCalibClosure)

## endpath
