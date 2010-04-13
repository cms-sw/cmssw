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
process.GlobalTag.globaltag = 'MC_36Y_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre5_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        # official RelVal 360pre5 single Photons pt=10GeV
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V3-v1/0010/18B0BB81-493E-DF11-8104-003048679294.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V3-v1/0009/1CA1E67D-BA3D-DF11-8736-001A92810ABA.root'
         

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre5 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0010/50398D8F-493E-DF11-B782-001A92810A98.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/FC832903-BC3D-DF11-BEC2-003048678FAE.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/FC682604-BB3D-DF11-9071-003048678BB2.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/F4FFE678-BA3D-DF11-A9D8-001A928116B0.root'

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



