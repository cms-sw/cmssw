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
[
       
'/store/relval/CMSSW_3_1_0_pre4/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v1/0001/189AF476-A516-DE11-8357-001A92810AA8.root',
       
'/store/relval/CMSSW_3_1_0_pre4/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v1/0001/3E0A9B94-6916-DE11-8393-003048678D52.root',
       
'/store/relval/CMSSW_3_1_0_pre4/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v1/0002/BC2D1B56-FE16-DE11-9048-0018F3D095EC.root' 
]    ),
    secondaryFileNames = cms.untracked.vstring()
)


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
