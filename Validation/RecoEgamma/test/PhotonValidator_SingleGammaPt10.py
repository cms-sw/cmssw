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
process.GlobalTag.globaltag = 'MC_38Y_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal383_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 383 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0022/E697FF38-EEBF-DF11-9210-0026189437F9.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0022/A271D36C-E7BF-DF11-9733-00248C0BE012.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0022/9C991A71-E8BF-DF11-AA6E-0018F3D095FA.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 383 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/EA8512EF-E7BF-DF11-8818-001A92810ADC.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/62F71637-EEBF-DF11-8752-003048678E6E.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/3E3D026D-E7BF-DF11-8DE1-0018F3D096CA.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/0ABB5470-E8BF-DF11-AD92-001A92811702.root'

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



