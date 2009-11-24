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
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_SingleGammaFlatPt10_100.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 340pre6 single Photons pt=10GeV

# official RelVal 340pre6 single Photons pt=35GeV            


# official RelVal 340pre6 single Photons Flat pt 10-100GeV


        
# official RelVal 340pre6 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/1872ADF9-48D7-DE11-8939-002354EF3BCE.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0001/F8C4B01B-BAD6-DE11-AAFA-00261894389D.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0001/EC09B8FD-BAD6-DE11-8A70-003048678BAE.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0001/7A30CB11-BCD6-DE11-8D36-002618943856.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0001/426788E5-B8D6-DE11-99B0-0026189438E0.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V14-v1/0001/2E91FA6B-B9D6-DE11-843E-0030486792BA.root'
# official RelVal 340pre6 GammaJets_Pt_80_120

# official RelVal 340pre6 QCD_Pt_80_120

 
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 340pre6 single Photons pt=10GeV    

    
# official RelVal 340pre6 single Photons pt=35GeV

# official RelVal 340pre6 single Photons Flat pt 10-100GeV


# official RelVal 340pre6 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/3EF28EFE-48D7-DE11-92A8-00261894393F.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/F460FA6C-B9D6-DE11-AC67-00248C0BE018.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/D4BE5FFF-BAD6-DE11-9483-00304867916E.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/B8F95DE4-B8D6-DE11-8D53-0030486792BA.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/ACA076FC-BAD6-DE11-ABE7-003048678C3A.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/98454BE4-B8D6-DE11-AA3C-0026189438EF.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/90E11011-BAD6-DE11-B0F0-003048D3C010.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/84CC1A6A-B9D6-DE11-9DB5-003048678B7E.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/6C8907E3-B8D6-DE11-BBAE-00261894398B.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/68CEFE84-BBD6-DE11-8674-003048678AC0.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/405C426A-B9D6-DE11-B66B-0026189437F0.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/3CEC0A6B-B9D6-DE11-A643-003048678F78.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0001/22CBC580-BAD6-DE11-BB6A-003048678BE6.root'




        
# official RelVal 340pre6 GammaJets_Pt_80_120

# official RelVal 340pre6 QCD_Pt_80_120

    
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



