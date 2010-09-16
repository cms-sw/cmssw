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
process.GlobalTag.globaltag = 'START38_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal383_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 383 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0022/DAB68168-F2BF-DF11-89AE-0018F3D096EC.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0022/AEF525A9-E1BF-DF11-B11A-00248C0BE005.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0022/AEAB2FDC-F3BF-DF11-A49E-002618943856.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0022/96D05423-EEBF-DF11-8FED-002618943946.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0021/B4C1125E-96BF-DF11-B2BD-00248C0BE01E.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0021/462138CD-92BF-DF11-93AF-003048D15DCA.root'

    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 383 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/D4A9D45F-29C0-DF11-A7D8-0018F3D096FE.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/B25B3B6E-F2BF-DF11-AC25-0018F3D09620.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/A21B30DD-F2BF-DF11-89B4-002618943832.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/7001F81A-EEBF-DF11-8B05-002618943984.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/50D55921-E1BF-DF11-936F-002618FDA262.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/34825C5D-F3BF-DF11-ABD3-0018F3D095F2.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/206456E2-F1BF-DF11-8BE9-0026189438E7.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/DE6B9060-92BF-DF11-933E-0026189438B0.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/8A39F7DC-94BF-DF11-875D-0026189438FD.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/503EC483-9CBF-DF11-A3D4-003048678FEA.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/229DFE48-8FBF-DF11-8E29-003048678C26.root',
        '/store/relval/CMSSW_3_8_3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/0C8C8DE4-95BF-DF11-B9A5-0018F3D09634.root'

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
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
