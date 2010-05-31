
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
process.GlobalTag.globaltag = 'START37_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0026/B428CB0F-8F69-DF11-90D3-002618943964.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/EC2A9079-3569-DF11-80D1-002618943886.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/AEC0CF76-3769-DF11-933E-00261894395F.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/8A215E8B-3969-DF11-BC7A-003048678E92.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/6EE21996-3C69-DF11-9F5F-00304867C29C.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/60E69877-3669-DF11-B464-002618943918.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/5C5F8602-3A69-DF11-92DB-002618943884.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/50B4BEF7-3669-DF11-BEFC-002618943967.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0026/E60473EB-8E69-DF11-8467-002618943964.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/FAFF9D86-3769-DF11-847A-002354EF3BDA.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/EE196E6D-3669-DF11-9715-003048D3C010.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/B03858FE-3969-DF11-87C0-002354EF3BDA.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/9AEC2185-3769-DF11-A825-001A92971B68.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/985E3F77-3569-DF11-B485-0018F3D09648.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/94112A26-3D69-DF11-B110-003048678FE0.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/7ABC2CE9-3669-DF11-B2A5-00261894382D.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/749C1786-3969-DF11-A18D-0018F3D095EC.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/72C49C02-3A69-DF11-899C-0026189438AC.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/4EF68393-3B69-DF11-ADE5-002618943843.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/4A707E5C-3469-DF11-AD3E-00304867C1BC.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/2CE7468D-3769-DF11-AE03-0026189438DA.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/2858D37C-3569-DF11-AC0B-0018F3D09686.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/24067214-3C69-DF11-92EC-00248C55CC9D.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/165B05E1-3569-DF11-BCAB-002354EF3BDA.root',
        '/store/relval/CMSSW_3_7_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/0A23C97E-3669-DF11-8EBD-003048678FE0.root'

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


