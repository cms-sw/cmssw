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
photonValidation.OutputFileName = 'PhotonValidationRelVal354_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 354 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0004/D48A6E41-A82B-DF11-ABA9-0017313F02F2.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0004/C2EFAB03-2D2C-DF11-B914-002618943985.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0004/9E196A44-A62B-DF11-BBB5-00248C0BE012.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0004/92571321-A82B-DF11-BB02-001A92811742.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0004/309C04D2-A52B-DF11-83A7-001A92810AEC.root'
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 354 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/E47A96CB-A52B-DF11-8928-001A92971BB8.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/DCA8C63B-A82B-DF11-8CCE-0018F3D096FE.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/DAE870FB-2C2C-DF11-B7BF-0026189438FE.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/AC4E095D-A62B-DF11-AA1A-003048678C06.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/A46B05A9-AD2B-DF11-A44C-0018F3D0966C.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/A29AD031-A52B-DF11-AB1C-001BFCDBD184.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/7A670CC9-A52B-DF11-8B41-00261894396D.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/5A0CB6CB-A62B-DF11-82F2-003048678C06.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/400F7B2C-A72B-DF11-B07F-00261894390A.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/32D21EC0-A62B-DF11-948E-001A928116EA.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/2002EE46-A62B-DF11-842A-003048678EE2.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/103AA82F-A92B-DF11-8014-0018F3D09698.root',
        '/store/relval/CMSSW_3_5_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0004/0CDE883C-A82B-DF11-962D-001A928116D8.root'

    
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
