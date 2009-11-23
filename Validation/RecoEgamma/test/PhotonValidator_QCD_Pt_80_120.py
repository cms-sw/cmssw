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
process.GlobalTag.globaltag = 'MC_3XY_V12::All'

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
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_GammaJets_Pt_80_120.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 340pre5 single Photons pt=10GeV

# official RelVal 340pre5 single Photons pt=35GeV            

# official RelVal 340pre5 single Photons Flat pt 10-100GeV


        
# official RelVal 340pre5 RelValH130GGgluonfusion
        
# official RelVal 340pre5 GammaJets_Pt_80_120

# official RelVal 340pre5 QCD_Pt_80_120
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/E2E45161-F7CB-DE11-B0E3-001D09F29849.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/D2929B7C-08CC-DE11-A8F3-001D09F231C9.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/BCA5EDFE-FFCB-DE11-84F7-0030487D1BCC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/B808C11E-F4CB-DE11-94C8-001D09F24D4E.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/72182DCD-F6CB-DE11-9DEF-0030487A1FEC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/6A0C4C36-F6CB-DE11-A83F-00304879FA4A.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/4E81D612-F5CB-DE11-BF67-001D09F2437B.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/483818AC-F5CB-DE11-A929-001D09F29169.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/12F39A01-F8CB-DE11-964F-001D09F252F3.root'


    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 340pre5 single Photons pt=10GeV    

    
# official RelVal 340pre5 single Photons pt=35GeV

# official RelVal 340pre5 single Photons Flat pt 10-100GeV


# official RelVal 340pre5 RelValH130GGgluonfusion


# official RelVal 340pre5 GammaJets_Pt_80_120

# official RelVal 340pre5 QCD_Pt_80_120

        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/FC5F4060-F7CB-DE11-BC2B-0030487D1BCC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/E03F2536-F6CB-DE11-B62A-001D09F25401.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/D899D086-F3CB-DE11-885B-001D09F29524.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/D05E0993-F6CB-DE11-9E19-0030487A18A4.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/C618AF80-08CC-DE11-916A-001D09F251CC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/B06BED35-F6CB-DE11-ACED-0030487D0D3A.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/9824C51E-F4CB-DE11-8DFF-001D09F295FB.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/94B090E9-F8CB-DE11-953B-0030487A1FEC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/8E1327BB-F7CB-DE11-893E-0030487A18F2.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/8CFDD6A7-F5CB-DE11-8B1A-0030487C6062.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/883D842B-F7CB-DE11-9EE5-001D09F28755.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/845EFCB1-F4CB-DE11-BF33-0019B9F72BFF.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/768152D2-F5CB-DE11-9CB7-0030487C6090.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/70D92D7E-F4CB-DE11-A878-001D09F2437B.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/6C6AA1FE-FFCB-DE11-98C9-0030487A1FEC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/6850E042-F5CB-DE11-9512-001D09F29849.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/66EAE711-F5CB-DE11-AEAF-001D09F253FC.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/50B6EFCA-F6CB-DE11-8CD6-0030487A3C9A.root'
    
    )
 )


photonPostprocessing.rBin = 48

## For single gamma fla pt =10-150
#photonValidation.eMax  = 300
#photonValidation.etMax = 300
#photonValidation.etScale = 0.10


## For single gamma pt =10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonPostprocessing.eMax  = 100
#photonPostprocessing.etMax = 50


## For single gamma pt = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.dCotCutOn = False
#photonValidation.dCotCutValue = 0.15
#photonValidation.likelihoodCut = 0.90




## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



