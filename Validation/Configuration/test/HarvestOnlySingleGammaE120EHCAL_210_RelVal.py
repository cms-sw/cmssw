import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SingleGamma results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_0_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/142F34A8-0961-DD11-8F52-0017312A21EB.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/50982484-0B61-DD11-9E7D-001A9281172C.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/8246FBA1-0961-DD11-AC1B-003048727005.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/B4FA94CC-0E61-DD11-B85E-001A92810AA2.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/F8A8B6AE-0961-DD11-8ABD-0018F3D096BE.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleGammaE120EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0002/2EB4D781-6A61-DD11-B492-003048767FAB.root'
    )
)

process.DQMStore.collateHistograms = False

#process.DQMStore.referenceFileName = cms.untracked.string(
#    'DQM_R000000001__ConverterTester__CMSSW_2_1_2__SIM_SingleGammaE120EHCAL_RelVal.root')


process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_2_1_0_REFERENCE/SingleGammaE120EHCAL'
#process.dqmSaver.referenceHandling = cms.untracked.string('skip')
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
process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)


