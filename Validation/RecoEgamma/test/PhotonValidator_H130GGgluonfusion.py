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
process.GlobalTag.globaltag = 'START36_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre4_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0002/C0D53CE9-1638-DF11-8208-0030487A195C.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0001/C6568825-8B37-DF11-9BCF-0030487A195C.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0001/AEA178F2-8B37-DF11-9DF5-0030487CD7EA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0001/969D8E89-8C37-DF11-A703-0030487A3DE0.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0001/9429B0B6-8B37-DF11-8BCD-001D09F24F65.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0001/444894BC-8C37-DF11-B5B8-0030487C90D4.root'
  
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 360pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/DC6F1631-1738-DF11-BBAD-0030487CD840.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/FE2CEBFF-8A37-DF11-BA02-0030487A3232.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/F434B424-8B37-DF11-8B11-0030487A1990.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/DA5CFAB1-8B37-DF11-B3F9-001D09F2516D.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/D012C3F6-8B37-DF11-A16B-0030487C7828.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/CA8CD3B7-8C37-DF11-ADB0-0030487D05B0.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/C6095FB3-8C37-DF11-A442-0030487CD7C0.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/B29E85B7-8B37-DF11-9E22-001617C3B69C.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/96708751-8A37-DF11-95C5-0030487C6088.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/50B2BEB3-8B37-DF11-AD02-001D09F2525D.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/288005E5-8B37-DF11-9F1F-0030487CD812.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/1A026052-8C37-DF11-A599-0030487CAEAC.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0001/1887C681-8C37-DF11-A62B-0030487CD76A.root'    
  
    
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
