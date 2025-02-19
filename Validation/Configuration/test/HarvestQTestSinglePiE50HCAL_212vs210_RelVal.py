import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SinglePiPt60 results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_2_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/329EFA5E-5376-DD11-A185-001A92971BDA.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/A8291840-5376-DD11-BCD9-00304876A0DB.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/BA3657B9-8F76-DD11-925E-001731A28A31.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/F8B23A3A-5376-DD11-BA91-003048767D35.root'
    )
)
process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('Validation/Configuration/data/QTHcal.xml'),
    #QualityTestPrescaler = cms.untracked.int32(1)
    reportThreshold = cms.untracked.string('black'),
    prescaleFactor = cms.untracked.int32(1),
    #qtList = cms.untracked.FileInPath('file.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndJob=cms.untracked.bool(True),
    testInEventloop=cms.untracked.bool(False),
    #Adding a new parameter to avoid running qtest multiple times
    qtestOnEndLumi=cms.untracked.bool(False)
)
process.DQMStore.collateHistograms = False

process.DQMStore.referenceFileName = cms.untracked.string(
    '/castor/cern.ch/user/g/gbenelli/SimValidation/DQMReferences/DQM_V0001_R000000001__SimRelVal__CMSSW_2_1_0_REFERENCE__SinglePiE50HCAL.root')

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_212vs210/SinglePiE50HCAL'
process.dqmSaver.referenceHandling = cms.untracked.string('skip')

process.DQMStore.verbose=3

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)

#Adding DQMFileSaver to the message logger configuration
process.MessageLogger.categories.append('DQMFileSaver')
process.MessageLogger.cout.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.MessageLogger.cerr.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.p1 = cms.Path(process.EDMtoMEConverter*process.qTester*process.dqmSaver)


