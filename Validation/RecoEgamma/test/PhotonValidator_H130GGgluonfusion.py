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
process.GlobalTag.globaltag = 'MC_3XY_V24::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal353_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 353 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0002/CC716EFA-3B28-DF11-A168-002618943907.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0001/EE43E39E-9827-DF11-934D-0026189437FD.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0001/E219EDAB-9927-DF11-A2B3-00304867926C.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0001/4631FE98-9827-DF11-BE41-0026189437FD.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0001/049CBC24-9927-DF11-9664-00304867904E.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 353 RelValH130GGgluonfusion


        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/C6361FF9-3B28-DF11-AB41-00261894398D.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/E446B4A6-9927-DF11-B61F-003048678E24.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/D628D124-9927-DF11-AE40-00261894392F.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/C6C6B896-9827-DF11-B242-003048679296.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/C43BB3A5-9927-DF11-8BC9-003048679296.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/C2DF8C1B-9927-DF11-B380-003048678A6C.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/B8A3579C-9827-DF11-A81C-0026189437F9.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/B61F9C9C-9827-DF11-B9CB-0030486792AC.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/9692CA98-9827-DF11-AE0E-00261894384F.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/6AA8BD9D-9827-DF11-91DA-003048679000.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/4A4B2620-9927-DF11-B082-002618943954.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/1473DB97-9827-DF11-BD4C-0026189438A5.root',
        '/store/relval/CMSSW_3_5_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/1018D110-9827-DF11-B762-00261894398D.root'
    
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
