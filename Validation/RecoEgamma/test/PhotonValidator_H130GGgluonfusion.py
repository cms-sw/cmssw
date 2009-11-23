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
photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_QCD_Pt_80_120.root'

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

        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0002/A452EBC9-10CC-DE11-BA57-0030487C6062.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/8061E6FD-9CCB-DE11-93EF-001D09F24498.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/64BD9636-9CCB-DE11-A7C8-001D09F24600.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/52BFE7C1-9DCB-DE11-A10F-001D09F23944.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/50DDE141-9ECB-DE11-ABA4-001D09F24498.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V11-v1/0001/0CFA1B64-A0CB-DE11-9750-001D09F24F1F.root'


# official RelVal 340pre5 GammaJets_Pt_80_120

# official RelVal 340pre5 QCD_Pt_80_120

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 340pre5 single Photons pt=10GeV    

    
# official RelVal 340pre5 single Photons pt=35GeV

# official RelVal 340pre5 single Photons Flat pt 10-100GeV


# official RelVal 340pre5 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/E2E391E4-F8CB-DE11-AF33-0030487A3C9A.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0002/4A24CCCB-10CC-DE11-B042-0030487D0D3A.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/FCA6C6D0-9BCB-DE11-8657-001D09F231C9.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/FA94DB86-9ECB-DE11-95AF-001D09F2AF1E.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/E00A1A6B-9DCB-DE11-BFE4-001D09F24024.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/D82E5E41-9ECB-DE11-B323-0019B9F72BFF.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/942521F4-9DCB-DE11-AFF4-0030487A1990.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/7ECA0235-9CCB-DE11-A856-0030487C6062.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/7EC87DB2-9BCB-DE11-A570-001D09F24493.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/700BD9FD-9CCB-DE11-8BAF-001D09F2AD7F.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/5CCA2B3F-9FCB-DE11-B0BC-001D09F250AF.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/46B6EA9C-9DCB-DE11-B49D-001D09F2AD7F.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/1E6D85CE-9CCB-DE11-A433-001D09F2423B.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V11-v1/0001/0C9D83C0-9DCB-DE11-8359-001D09F24489.root'

# official RelVal 340pre5 GammaJets_Pt_80_120

# official RelVal 340pre5 QCD_Pt_80_120

    
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



