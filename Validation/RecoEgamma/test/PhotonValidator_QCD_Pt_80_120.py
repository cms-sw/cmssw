
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
process.GlobalTag.globaltag = 'START38_V8::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal381_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 381 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0011/8C85C9BF-31A2-DF11-9C55-002618943861.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/CE921445-D1A1-DF11-AB05-00261894394D.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/B897E3B8-D1A1-DF11-B548-001A92811708.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/B83757A6-D0A1-DF11-84B4-001731EF61B4.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/820809A8-D2A1-DF11-8269-002618943947.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/807DFB3E-D1A1-DF11-9930-001A9281173C.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/0E32C730-D6A1-DF11-AA54-003048679046.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0010/00A2289B-D0A1-DF11-AED7-00304867BFC6.root'
      ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 381 QCD_Pt_80_120

        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0011/32B7AC18-32A2-DF11-A7A8-00261894394A.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/C875F08C-D0A1-DF11-927B-001A92810AA2.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/C03342F9-CFA1-DF11-BEE6-0026189438F2.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/AC50EF2D-D6A1-DF11-97EA-003048679162.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/A4549CA2-D1A1-DF11-AC9B-0030486791BA.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/9AE08315-D2A1-DF11-8D35-001A92810AD8.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/90EE789E-D2A1-DF11-A1B9-002618943956.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/7C2B24A8-D1A1-DF11-B768-003048678B20.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/72825A99-D0A1-DF11-B85E-001A928116DA.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/62A4E87A-D6A1-DF11-BCC9-002618943967.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/5A335D11-D1A1-DF11-920F-0026189438C0.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/52189015-D1A1-DF11-84C6-0018F3D096EC.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/50AAD30F-D1A1-DF11-AC67-001A92811708.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/4A517C1B-D1A1-DF11-B74B-00261894392B.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/30F735FF-D4A1-DF11-B37E-003048678B0A.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/2E15F997-D0A1-DF11-B481-002618943983.root',
        '/store/relval/CMSSW_3_8_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/140C322B-D2A1-DF11-BEE7-001A928116DE.root'
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


