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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre4_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre4 single Photons pt=35GeV            
 '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V3-v1/0001/E227AEAE-8A37-DF11-9B3D-0030487A3232.root',
 '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V3-v1/0001/C66EF74F-8937-DF11-BA6D-0030487CD906.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0002/3E65DF2E-1738-DF11-9D68-0030487C608C.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/A4EB1F0E-8937-DF11-9E6F-0030487CD6B4.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/94FB9951-8A37-DF11-8CF2-0030487A3232.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/3C784393-8937-DF11-AAAB-0030487A1FEC.root'

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
