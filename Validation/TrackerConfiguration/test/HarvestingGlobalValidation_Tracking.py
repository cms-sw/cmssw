import FWCore.ParameterSet.Config as cms

import os



process = cms.Process("EDMtoMEConvert")
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Validation.Configuration.postValidation_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/1EF32A82-57E2-DD11-A475-000423D6B444.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/28804479-4BE2-DD11-A32D-000423D98EA8.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/44067402-4BE2-DD11-BD29-0030487C6090.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/6CCDD56F-4BE2-DD11-9078-001D09F27067.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/888637D0-4AE2-DD11-B5AD-000423D6CA6E.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/B6E15573-4BE2-DD11-B754-001D09F24E39.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/B8148F8C-4BE2-DD11-B0C8-001D09F28D54.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/D4855570-4BE2-DD11-8793-000423D98930.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = "/"+os.environ["CMSSW_VERSION"]+"/RelVal/Validation"
process.DQMStore.verbose=3

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)

# Other statements

#Adding DQMFileSaver to the message logger configuration
process.MessageLogger.categories.append('DQMFileSaver')
process.MessageLogger.cout.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.MessageLogger.cerr.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )

process.post_validation= cms.Path(process.postValidation)
process.EDMtoMEconv_and_saver= cms.Path(process.EDMtoMEConverter*process.dqmSaver)

process.schedule = cms.Schedule(process.post_validation,process.EDMtoMEconv_and_saver)


for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""
