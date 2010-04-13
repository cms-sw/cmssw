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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre5_SingleGammaPt35TEST.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre5 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V3-v1/0010/741C66A5-493E-DF11-A09F-001A92971B04.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V3-v1/0009/AA8E0935-B53D-DF11-B954-0018F3D096A0.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre5 single Photons pt=35GeV
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0010/708B76A2-493E-DF11-B2B4-001A92810AAA.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/301BD376-B53D-DF11-B8C3-002618943950.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/14CABF57-B43D-DF11-A6E5-003048D15D22.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0009/12E66A19-B53D-DF11-996E-0018F3D09644.root'


    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
