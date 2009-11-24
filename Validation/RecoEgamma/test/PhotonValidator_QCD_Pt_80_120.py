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
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_H130GGgluonfusion.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_GammaJets_Pt_80_120.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal340pre6_QCD_Pt_80_120.root'

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
        
# official RelVal 340pre6 GammaJets_Pt_80_120

# official RelVal 340pre6 QCD_Pt_80_120

        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/D0C13CEF-48D7-DE11-B2CB-00261894391D.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/CC63CFCA-19D7-DE11-AD3C-002618943843.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/BC4A175B-10D7-DE11-8876-003048678B76.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/B82606E5-14D7-DE11-90DF-00304867901A.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/9836F6C4-13D7-DE11-9CC5-00261894390B.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/863B3335-6ED7-DE11-A9D3-00261894393B.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/826889BD-11D7-DE11-983F-003048678A6A.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/441A1FE8-14D7-DE11-B01E-001A92810ADC.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/40B91D15-17D7-DE11-BFB6-0018F3D096BC.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0002/245B9E87-66D7-DE11-BFAD-002618943865.root'
          

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 340pre6 single Photons pt=10GeV    

    
# official RelVal 340pre6 single Photons pt=35GeV

# official RelVal 340pre6 single Photons Flat pt 10-100GeV


# official RelVal 340pre6 RelValH130GGgluonfusion


# official RelVal 340pre6 GammaJets_Pt_80_120

# official RelVal 340pre6 QCD_Pt_80_120

        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/F6935E63-15D7-DE11-8750-001731AF6865.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/F43FFF85-66D7-DE11-B131-002354EF3BDD.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/C610C335-6ED7-DE11-9AE4-002618943935.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/BE764CE0-13D7-DE11-8DF8-001A92810AEC.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/A2ADC912-17D7-DE11-BBF4-0030486790FE.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/86B9C47C-17D7-DE11-A7F6-0030486791F2.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/849EA06B-23D7-DE11-8144-001731230E47.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/744E0083-16D7-DE11-9196-001731AF66A7.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/6C058B88-13D7-DE11-A5D2-0018F3D09690.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/66E42D7F-19D7-DE11-8EEC-002618943843.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/460AD12A-10D7-DE11-8207-003048678A6A.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/42416076-11D7-DE11-BBB6-002618943900.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/42133015-12D7-DE11-AF6F-0018F3D0960A.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/3EE17025-10D7-DE11-80D6-003048678AFA.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/3E947EE8-13D7-DE11-8C8B-001A92810AF2.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/2865DF56-11D7-DE11-A36B-003048678B06.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/164D93E7-14D7-DE11-86E5-003048678BE6.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/14321DE9-14D7-DE11-B558-0026189438AD.root',
        '/store/relval/CMSSW_3_4_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0002/0EF60A2B-13D7-DE11-9D46-0026189438F2.root'

     
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



