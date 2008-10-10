import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/22CEB9BB-8F76-DD11-A820-003048767D3D.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/2466473F-5376-DD11-8590-001731AF685F.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/2EE56D3A-5376-DD11-8F83-001731AF6787.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/7CADC93B-5376-DD11-9DF4-003048767DEF.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/AAE4133C-5376-DD11-9814-00304875AA71.root',
    '/store/relval/CMSSW_2_1_2/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v2/0001/C8D00A40-5376-DD11-817A-001731AF67BF.root'
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
    '/castor/cern.ch/user/g/gbenelli/SimValidation/DQMReferences/DQM_V0001_R000000001__SimRelVal__CMSSW_2_1_0_REFERENCE__SingleGammaE120EHCAL.root')

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_212vs210/SingleGammaE120EHCAL'
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


