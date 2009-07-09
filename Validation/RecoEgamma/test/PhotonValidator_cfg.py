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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V1::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
photonValidation.OutputMEsInRootFile = True
#photonValidation.OutputFileName = 'PhotonValidationRelVal311_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal311_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal311_SingleGammaFlatPt10_100.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal311_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal311_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal311_QCD_Pt_50_80.root'

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 311 single Photons pt=10GeV
#'/store/relval/CMSSW_3_1_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V2-v1/0002/A0BFDEDF-776B-DE11-9054-000423D98C20.root'


# official RelVal 311 single Photons pt=35GeV            

# '/store/relval/CMSSW_3_1_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V2-v1/0002/ACFD3872-646B-DE11-98AA-000423D94700.root',
# '/store/relval/CMSSW_3_1_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V2-v1/0002/98DB60E5-D66B-DE11-BDAB-000423D9853C.root'
  
# official RelVal 311 single Photons Flat pt 10-100GeV


        
# official RelVal 311 RelValH130GGgluonfusion

 '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/CCA710E9-C86B-DE11-BA96-000423D986C4.root',
 '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/C2264DF5-C46B-DE11-8F51-0030487A3C9A.root',
 '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/9C50A901-C76B-DE11-BB52-000423D174FE.root',
 '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/643B6BEF-BE6B-DE11-9740-001D09F232B9.root',
 '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/421F8DD4-E16B-DE11-AF4C-000423D98844.root'


        
# official RelVal 311 GammaJets_Pt_80_120

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 311 single Photons pt=10GeV    
#'/store/relval/CMSSW_3_1_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/AE0D57A3-766B-DE11-B3AD-001D09F282F5.root',
#'/store/relval/CMSSW_3_1_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/5E3F9F50-776B-DE11-8B48-000423D99F1E.root',
#'/store/relval/CMSSW_3_1_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0AB75C56-D66B-DE11-B631-001D09F2514F.root'

        
 
# official RelVal 311 single Photons pt=35GeV

#        '/store/relval/CMSSW_3_1_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EEE51923-626B-DE11-8143-001D09F2960F.root',
#        '/store/relval/CMSSW_3_1_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/CE88E607-D86B-DE11-85BD-000423D99AAA.root',
#        '/store/relval/CMSSW_3_1_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/C0B73817-606B-DE11-8C68-001D09F24498.root'
        
        
# official RelVal 311 single Photons Flat pt 10-100GeV


# official RelVal 311 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/EC68E778-BD6B-DE11-A97F-001D09F2523A.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/DE9EBEFD-C56B-DE11-8B8C-000423D98B6C.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/D4F18A07-C56B-DE11-9FA0-0030487A1990.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/B0605C4C-BE6B-DE11-A4DB-001D09F25041.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/A07DBA05-C76B-DE11-A01E-000423D951D4.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/924B7CF2-C66B-DE11-983D-000423D98AF0.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/86EE66C4-BE6B-DE11-83EB-001D09F29169.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/80C0360B-D76B-DE11-92A6-001D09F2514F.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/7ABD0F3F-BF6B-DE11-9650-001D09F24498.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6421EEE1-C76B-DE11-8640-0019B9F707D8.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6068CF58-C86B-DE11-8BF8-000423D944DC.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/2062D244-C56B-DE11-81AD-000423D944DC.root',
        '/store/relval/CMSSW_3_1_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/0E76EE4D-C16B-DE11-B27A-000423D6B358.root'



# official RelVal 311 GammaJets_Pt_80_120

    
    )
 )



## For single gamma fla pt =10-150
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 300
#photonValidation.etMax = 300
#photonValidation.etScale = 0.10
#photonValidation.signal = True

## For single gamma pt =10
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For single gamma pt = 35
#photonValidation.minPhoEtCut = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For gam Jet and higgs
photonValidation.minPhoEtCut = 20
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonValidation.signal = True




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.dqmStoreStats)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



