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
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/1204CF8F-5876-DD11-B46E-001A92810AD2.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/2C878E2E-5876-DD11-A341-001A92810AEA.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/822F6EB9-5876-DD11-A21F-003048767DEF.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/96ACD558-5776-DD11-8C31-001731AF664F.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/AA5D2FBB-8F76-DD11-9419-001731AF6B77.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/C050E5C9-5876-DD11-A6E5-00304875A7B5.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/CA3E2AAA-5976-DD11-908E-001A92971B38.root'
    )
)
process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('Validation/Configuration/data/QTEcalHcal.xml'),
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
    '/castor/cern.ch/user/g/gbenelli/SimValidation/DQMReferences/DQM_V0001_R000000001__SimRelVal__CMSSW_2_1_0_REFERENCE__SinglePiPt60EHCAL.root')

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_212vs210/SinglePiPt60EHCAL'
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


