
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
photonValidation.OutputFileName = 'PhotonValidationRelVal341_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 341 QCD_Pt_80_120

        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D8AE64A2-8FED-DE11-B37B-000423D99658.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D814474A-90ED-DE11-A54F-001D09F2A49C.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D2235DA4-8FED-DE11-A5FE-001D09F241F0.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/C2371774-B6ED-DE11-9292-0030487C5CFA.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/B84A6EE6-8EED-DE11-9733-001D09F29321.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/58E9DA6E-8DED-DE11-91C8-003048D3756A.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/48D81FB5-91ED-DE11-9BFA-001617C3B5E4.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/0AB1FC24-91ED-DE11-8A56-001D09F231B0.root'          

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 341 QCD_Pt_80_120

        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/FE40E040-90ED-DE11-BA4C-001D09F24DDF.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/E24005DF-8EED-DE11-8C40-001D09F25393.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/E21C32BA-91ED-DE11-A68E-001D09F253D4.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/A2C66E6D-8DED-DE11-B883-001D09F24DDF.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/9C97CE7D-8EED-DE11-A6DB-001D09F2A690.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/9815757C-B5ED-DE11-BE11-000423D6CA72.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/8AF4D09D-8FED-DE11-B276-001617C3B66C.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/6893514A-90ED-DE11-870E-001D09F251BD.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/5C98111F-8FED-DE11-9730-001D09F232B9.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/5A4A9044-90ED-DE11-A455-001D09F2525D.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/523D7026-8EED-DE11-AD4C-001617E30D12.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/4C0A8D9E-91ED-DE11-8E70-001D09F23D1D.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/4620646B-8DED-DE11-BABC-001D09F244BB.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/30906DFA-90ED-DE11-9BBD-000423D98950.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/28D86116-91ED-DE11-98CC-000423D94A04.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/1866D9A4-8FED-DE11-9180-001D09F29619.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/0EF698A0-8FED-DE11-830A-001D09F251BD.root',
        '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/046D6F15-91ED-DE11-9264-001D09F24399.root'

     
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


