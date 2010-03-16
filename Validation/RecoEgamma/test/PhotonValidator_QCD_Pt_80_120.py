
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
photonValidation.OutputFileName = 'PhotonValidationRelVal353_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 353 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/E62A31AC-3A28-DF11-B928-00261894383E.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/C291DD81-B227-DF11-91BD-00304867915A.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/A46D6989-B127-DF11-B84E-003048678B94.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/96FF4F82-B227-DF11-BD7D-00248C0BE005.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/94CF6DAF-B227-DF11-AA55-002618FDA211.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/4614BB3E-B227-DF11-8015-00304867BED8.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/2A0F7D84-B227-DF11-A298-001BFCDBD19E.root'

    ),

    secondaryFileNames = cms.untracked.vstring(

# official RelVal 353 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/F23D40BB-B227-DF11-855F-00261894380D.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/E40FB334-B227-DF11-ABE2-002618943870.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/D6EF8B2D-AF27-DF11-8542-003048679008.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/D0AE7CB7-B227-DF11-896F-00261894393E.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/CAA74C6D-B227-DF11-8A2D-002618943852.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/BE817B52-B227-DF11-94BE-002618943985.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/B4B26D4D-B227-DF11-95D2-00261894397E.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/80704989-B127-DF11-9340-002354EF3BDD.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/72672C3A-B227-DF11-B2CC-002618FDA211.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/66F7CD5F-B227-DF11-A531-002618943953.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/5A1E46B8-B227-DF11-BE48-00261894386A.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/5681EB5E-B227-DF11-8ED9-003048678B94.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/4C1EB8DF-3A28-DF11-801B-002618943907.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/24B8B289-B127-DF11-930F-003048678F84.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/1CA3A977-B227-DF11-B1EB-002618FDA211.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/0E86C86F-B227-DF11-A302-002618943885.root',
        '/store/relval/CMSSW_3_5_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/0AA7FE58-B227-DF11-9898-003048678FE0.root'

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


