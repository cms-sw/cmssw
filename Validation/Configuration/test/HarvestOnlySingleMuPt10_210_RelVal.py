import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            #Cut and paste here the SingleMuPt10 results of ~gbenelli/public/SimValidation/aSearchCLI.py --input="find dataset where dataset like *CMSSW_2_1_0_Special*"
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/0A2D1D45-0D61-DD11-937B-001BFCDBD154.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/52AEAAF2-0D61-DD11-B731-0017312B55FF.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/863B2059-0C61-DD11-AE8F-0018F3D0969A.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/C4800E2D-0D61-DD11-AD8A-001731AF6781.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0001/DE79A84D-0D61-DD11-854D-001A928116E6.root',
    '/store/relval/CMSSW_2_1_0/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/SpecialConditions_v1/0002/781EC7CE-6A61-DD11-9362-001A92971B74.root'
    )
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = '/SimRelVal/CMSSW_2_1_0_REFERENCE/SingleMuPt10'
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


