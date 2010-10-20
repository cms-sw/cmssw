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
photonValidation.OutputFileName = 'PhotonValidationRelVal390_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0050/82A4E1F9-FDD7-DF11-9432-001A928116EA.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0049/781C43DB-F7D7-DF11-A5BE-003048678B18.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0049/64F08F61-F6D7-DF11-8A55-002618943869.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0049/4C5F364F-F7D7-DF11-988F-003048678FD6.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0049/2AE23CCF-F6D7-DF11-893E-003048678DA2.root'
  
 
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/581722DB-F7D7-DF11-8C7D-001A928116C2.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/1EB915DA-F7D7-DF11-8B6B-00304867C16A.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/187B0A9C-39D8-DF11-B027-00261894394F.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/067912DB-F7D7-DF11-B445-00261894398B.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/FA5FFFD2-F6D7-DF11-B1E8-001A9281173C.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/E67DD0CF-F6D7-DF11-AF2A-002618943875.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/DC85ECCE-F6D7-DF11-8A37-003048678B06.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/D6F002D1-F6D7-DF11-8B57-0018F3D096C2.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/CEF6F85F-F6D7-DF11-B33B-001A92811732.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/C097264E-F7D7-DF11-8D68-002618943875.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/B023DC5E-F6D7-DF11-9D43-00304867BEC0.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/6250905D-F6D7-DF11-BB18-00261894393A.root',
        '/store/relval/CMSSW_3_9_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0049/306C73CC-F6D7-DF11-BA09-0030486792DE.root'



    )
 )


photonPostprocessing.rBin = 48
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
