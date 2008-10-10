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
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/24610807-1261-DD11-81ED-0018F3D095F6.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/A67D282E-1361-DD11-AF0F-001A92810ADC.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/B0595518-1161-DD11-9F27-003048767E49.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/BEDA4930-1161-DD11-BFF0-0018F3D09628.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/CC365A2D-1261-DD11-A932-001A92811720.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/D850F639-1161-DD11-B7A4-001A928116BA.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiPt60EHCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0002/26FF2FA5-6A61-DD11-B048-001A92971B08.root'
    )
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_2_1_0_REFERENCE/SinglePiPt60EHCAL'
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


