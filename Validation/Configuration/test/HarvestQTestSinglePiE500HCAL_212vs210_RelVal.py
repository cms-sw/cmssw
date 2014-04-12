import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SinglePiE500HCAL results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_2_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/02C309F0-5E76-DD11-9837-001731A28FCB.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/6488C3E6-5E76-DD11-9F0F-003048767653.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/64B268B9-8F76-DD11-9523-001731AF684D.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/78EA2414-5F76-DD11-A59B-0018F3D095EA.root',
    '/store/relval/CMSSW_2_1_2/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/A06AE439-5E76-DD11-A904-0017312B554B.root'
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
    '/castor/cern.ch/user/g/gbenelli/SimValidation/DQMReferences/DQM_V0001_R000000001__SimRelVal__CMSSW_2_1_0_REFERENCE__SinglePiE500HCAL.root')

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_212vs210/SinglePiE500HCAL'
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


