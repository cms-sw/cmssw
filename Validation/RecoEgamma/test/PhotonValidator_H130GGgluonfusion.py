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
process.GlobalTag.globaltag = 'MC_3XY_V14::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre2_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 350pre2 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0010/26A2620C-22EE-DE11-AA42-00261894396A.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/D0E410D7-85ED-DE11-88D9-002618943951.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/A01E4E98-87ED-DE11-A54B-002618943911.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/9C1BA413-86ED-DE11-8CC2-0026189438BD.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/3C104E15-87ED-DE11-953E-002618943880.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/00645142-85ED-DE11-86B4-002618943860.root'
    
 
    ),
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 350pre2 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0010/3420051A-22EE-DE11-B601-002618943842.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/E4F7947F-87ED-DE11-BA43-0026189438C0.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/C4A7600B-87ED-DE11-BF4C-002618943880.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/9C9C7DA0-85ED-DE11-B134-003048678A78.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/82ACCF02-86ED-DE11-B2F5-00304867916E.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/72489CAA-85ED-DE11-810E-003048678FE6.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/6CB72932-85ED-DE11-97DF-0026189437F8.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/54189103-86ED-DE11-BF8F-003048678B86.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/4E1686B8-84ED-DE11-816F-00304867900C.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/2E48C8F9-85ED-DE11-AF21-003048679182.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/1AD6C032-85ED-DE11-9FF9-00261894396C.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/142ED402-87ED-DE11-AFAD-00261894394A.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/12BD7BFC-85ED-DE11-8F5D-003048678B86.root'


    
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
