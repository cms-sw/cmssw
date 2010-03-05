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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal352_SingleGammaPt35.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 352 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V21-v1/0016/BC2A5E5F-D91E-DF11-8516-003048678B18.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V21-v1/0015/44771CE9-171E-DF11-A55D-0018F3D095F0.root'
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
    

# official RelVal 352 single Photons pt=35GeV
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/E4F91C59-D91E-DF11-AEDE-003048678B04.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/C409F0E4-171E-DF11-B411-003048679164.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/56777CE5-171E-DF11-8E5E-002618943967.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/40D405A4-1A1E-DF11-99C3-001731AF68ED.root'

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
