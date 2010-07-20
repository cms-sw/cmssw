import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("Validation.RecoEgamma.photonPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_38Y_V4::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre7_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre7 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V4-v1/0002/78677308-5686-DF11-AE7B-0030487CD17C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V4-v1/0002/4040A3E3-5486-DF11-8D97-0030487A1FEC.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V4-v1/0002/1237AC54-8086-DF11-8813-003048F1C58C.root'
 
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre7 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0002/DE92446F-5586-DF11-B841-0030487CD6D8.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0002/A0DDBE7F-5486-DF11-ADE0-0030487CD17C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0002/9E04E3E0-5486-DF11-9070-0030487C5CFA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0002/585D5E68-8086-DF11-A2FA-0030487CD7EA.root' 


    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt =10
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)



