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
photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre5_H130GGgluonfusion.root'
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

    
   '/store/relval/CMSSW_3_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V12-v1/0002/DE719CF1-F7CB-DE11-9699-001D09F24600.root',
   '/store/relval/CMSSW_3_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V12-v1/0001/C0CADB96-A1CB-DE11-8855-001D09F295FB.root'
 
# official RelVal 340pre5 single Photons Flat pt 10-100GeV


        
# official RelVal 340pre5 RelValH130GGgluonfusion
 
        
# official RelVal 340pre5 GammaJets_Pt_80_120

# official RelVal 340pre5 QCD_Pt_80_120

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 340pre5 single Photons pt=10GeV    



    
# official RelVal 340pre5 single Photons pt=35GeV
        '/store/relval/CMSSW_3_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V12-v1/0002/BA9400E8-F8CB-DE11-B97C-001D09F24763.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V12-v1/0001/CE8B8604-A1CB-DE11-8EF8-001D09F2915A.root',
        '/store/relval/CMSSW_3_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V12-v1/0001/B60AB195-A1CB-DE11-AF2A-0030487C6090.root'

# official RelVal 340pre5 single Photons Flat pt 10-100GeV


# official RelVal 340pre5 RelValH130GGgluonfusion

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
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15
#photonValidation.likelihoodCut = 0.90




## For gam Jet and higgs
#photonValidation.eMax  = 500
#photonValidation.etMax = 500
#photonPostprocessing.eMax  = 500
#photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


#process.p1 = cms.Path(process.photonValidation)
process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



