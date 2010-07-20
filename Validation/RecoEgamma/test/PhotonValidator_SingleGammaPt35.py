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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre7_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre7 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V4-v1/0002/3A47A307-2B86-DF11-B764-0030487A1990.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V4-v1/0001/D8A8984A-C685-DF11-9EAA-003048F01E88.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V4-v1/0001/9A1C1959-C585-DF11-BEEE-003048F1182E.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre7 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0002/9CC1AAE7-2A86-DF11-9911-0030487CD178.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0001/F2AD9359-C585-DF11-BB23-00304879EDEA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0001/B6ED3BB2-C585-DF11-96AA-0030487CD7EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V4-v1/0001/1C7B4353-C585-DF11-991A-0030487CD704.root'
        
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
