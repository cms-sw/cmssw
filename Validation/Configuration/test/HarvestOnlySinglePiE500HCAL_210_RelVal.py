import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SinglePiE500HCAL results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_0_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/984E694D-1861-DD11-B15A-001A92810AA8.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/A898AF4C-1961-DD11-9054-001731AF66BF.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/B80F4446-1861-DD11-9ABA-001731AF68AB.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/D2A717ED-1861-DD11-A6AF-001A92971B28.root',
    '/store/relval/CMSSW_2_1_0/RelValSinglePiE500HCAL/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0002/76CF4385-6A61-DD11-AB5C-0018F3D09678.root'
    )
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_2_1_0_REFERENCE/SinglePiE500HCAL'
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


