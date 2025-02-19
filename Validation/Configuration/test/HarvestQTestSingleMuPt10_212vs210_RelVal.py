import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SingleMuPt10 results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_2_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/2096DCC3-8F76-DD11-A0E8-001731AF68CF.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/2671D53F-5376-DD11-92B4-001A92971B68.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/32AC948E-5376-DD11-9970-0018F3D09704.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/7CD68948-5376-DD11-ADE2-003048756687.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/8A884941-5376-DD11-B429-001A92810A94.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/BC4A505A-5376-DD11-BD61-001A92971B90.root'
    )
)
process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('Validation/Configuration/data/QTTrackerMuon.xml'),
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
    '/castor/cern.ch/user/g/gbenelli/SimValidation/DQMReferences/DQM_V0001_R000000001__SimRelVal__CMSSW_2_1_0_REFERENCE__SingleMuPt10.root')

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_212vs210/SingleMuPt10'
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


