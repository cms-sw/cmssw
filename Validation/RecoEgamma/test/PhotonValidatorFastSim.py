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
process.GlobalTag.globaltag = 'START39_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390_H130GGgluonfusion_FastSim.root'

photonValidation.fastSim = True

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0051/64CAB669-3DD8-DF11-A8FB-0018F3D0965E.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/F42D66FB-EFD7-DF11-AECF-002618943972.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/DA4151DB-EED7-DF11-96A8-001A928116F4.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/BA1EBD66-EFD7-DF11-A579-0026189438AF.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/8E5FC6ED-EFD7-DF11-80EF-00261894390B.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/86D8D8FB-EFD7-DF11-A892-00304867C026.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/7AA42FF9-EFD7-DF11-A80A-002618943866.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/580513F5-EFD7-DF11-961A-001A928116C4.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/4AC8A1FC-EFD7-DF11-8C64-003048678D9A.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/463122F7-F1D7-DF11-8617-002618943924.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/34169CED-EFD7-DF11-9171-00261894394F.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/28B23D67-EFD7-DF11-918D-003048678E80.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/207BB263-EFD7-DF11-BCDE-00261894390C.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V2_FastSim-v1/0049/147985F8-EFD7-DF11-B8FC-001BFCDBD154.root'


    )
    )


photonPostprocessing.rBin = 48

## For single gamma pt =10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonPostprocessing.eMax  = 100
#photonPostprocessing.etMax = 50

## For single gamma pt = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.dCotCutOn = False
#photonValidation.dCotCutValue = 0.15


## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
