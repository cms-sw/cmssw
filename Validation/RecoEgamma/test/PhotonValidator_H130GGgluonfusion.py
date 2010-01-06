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
process.GlobalTag.globaltag = 'MC_31X_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal341_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 341 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/F62C7DB8-83ED-DE11-A33B-001D09F25041.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/CA2FE932-85ED-DE11-92BC-001D09F2525D.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/A6C37A6E-84ED-DE11-9741-003048D2C020.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/769D4A78-B5ED-DE11-8FDC-0030487C6062.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/56C5950C-83ED-DE11-BCFB-001D09F28755.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/184EEABC-83ED-DE11-B6F6-001D09F24D4E.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(



# official RelVal 341 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/FCF7FF05-83ED-DE11-9D9B-003048D2C1C4.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/E8F3D907-83ED-DE11-8E37-001D09F29321.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/E2F4FE26-85ED-DE11-8CC8-001D09F2AD7F.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/C86B8CB6-83ED-DE11-86B1-001D09F24E39.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/C6CF1FBC-83ED-DE11-891C-001D09F290CE.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/BAF6FD71-84ED-DE11-BCD0-001D09F23A20.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/A661F605-83ED-DE11-A13F-003048D37560.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/9A5320BE-83ED-DE11-93CE-001D09F24353.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/726F952B-85ED-DE11-AE0F-000423D6006E.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/60B6BB6C-84ED-DE11-89F9-001617E30E28.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/52A0200A-83ED-DE11-906A-001D09F2910A.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/4C3E29D6-B6ED-DE11-A1D9-0030487A18A4.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/3A7A37B8-83ED-DE11-962F-001D09F2546F.root',
        '/store/relval/CMSSW_3_4_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/322431B8-83ED-DE11-8B9C-001D09F27067.root'

    
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
