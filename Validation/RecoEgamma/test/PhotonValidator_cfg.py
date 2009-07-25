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
process.GlobalTag.globaltag = 'MC_31X_V3::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
#photonValidation.OutputFileName = 'PhotonValidationRelVal312_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal312_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal312_SingleGammaFlatPt10_100.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal312_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal312_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal312_QCD_Pt_50_80.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = 'PhotonValidationRelVal312_H130GGgluonfusion.root'
process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 312 single Photons pt=10GeV

#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V3-v1/0006/F0390882-5C78-DE11-9CAC-001D09F24600.root'

# official RelVal 312 single Photons pt=35GeV            
#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V3-v1/0007/123C78F5-9078-DE11-8BAD-001D09F23A61.root',
#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V3-v1/0006/E6C7ED95-4878-DE11-B082-000423D98BE8.root'
  
# official RelVal 312 single Photons Flat pt 10-100GeV


        
# official RelVal 312 RelValH130GGgluonfusion

'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/FC6C2AF4-E278-DE11-B2D1-001D09F23A07.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/CEA857A2-CC78-DE11-8073-000423D98EA8.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/BA7034AD-CC78-DE11-96E0-001D09F251E0.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/748489A8-CC78-DE11-991C-000423D99896.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/104E25AC-CC78-DE11-AE55-001D09F2447F.root'

        
# official RelVal 312 GammaJets_Pt_80_120

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 312 single Photons pt=10GeV    

#   '/store/relval/CMSSW_3_1_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0007/BA38AB4C-A878-DE11-90D7-000423D6CAF2.root',
#   '/store/relval/CMSSW_3_1_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/E4A63D01-5B78-DE11-AFC0-000423D95030.root',
#   '/store/relval/CMSSW_3_1_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/34B84B7C-5C78-DE11-A44E-000423D99658.root'
 
# official RelVal 312 single Photons pt=35GeV

#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0007/46413B52-9278-DE11-9BC3-000423D99CEE.root',
#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/D6842ADB-4678-DE11-A4A3-001D09F28F1B.root',
#'/store/relval/CMSSW_3_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0006/AA6869A4-4578-DE11-8F9C-000423D6CAF2.root'        

# official RelVal 312 single Photons Flat pt 10-100GeV


# official RelVal 312 RelValH130GGgluonfusion

'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/FCD194D7-CB78-DE11-9BC2-000423D8FA38.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/F812188A-CC78-DE11-87E1-001D09F24D4E.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/EC88A691-CC78-DE11-9B3E-001D09F24259.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/C67AB78C-CC78-DE11-90F8-001D09F28F11.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/AC6F95F9-E278-DE11-9836-001D09F24600.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/AC677AE3-CB78-DE11-AF27-001D09F2305C.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/A665C68A-CC78-DE11-B822-000423D990CC.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/7223B891-CC78-DE11-A938-001D09F244BB.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/68508D8F-CC78-DE11-A450-001D09F28EA3.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/520B9691-CC78-DE11-ABFC-0019B9F707D8.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/3A3F61DD-CB78-DE11-A8DB-001D09F24DDA.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/3A247192-CC78-DE11-B90B-0019B9F705A3.root',
'/store/relval/CMSSW_3_1_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/26CC858A-CC78-DE11-A679-001D09F28755.root'


# official RelVal 312 GammaJets_Pt_80_120

    
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
#photonValidation.minPhoEtCut = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20


## For gam Jet and higgs
#photonValidation.minPhoEtCut = 20
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



