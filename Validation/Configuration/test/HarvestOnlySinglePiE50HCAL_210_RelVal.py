import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SinglePiPt60 results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_0_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/40606BE4-0661-DD11-B624-001A928116F2.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/74645B20-0761-DD11-9A93-0018F3D09710.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/FE5E80D5-0761-DD11-A5C5-001A928116D4.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE50HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0002/00FF3E80-6A61-DD11-8BA8-00304876A153.root'
    )
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_2_1_0_REFERENCE/SinglePiE50HCAL'
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


