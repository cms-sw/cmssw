import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")
process.load("Validation.RecoEgamma.tpSelection_cfi")


process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
 input = cms.untracked.int32(10000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_QCD_Pt_50_80.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre6_H130GGgluonfusion.root'

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


        
        #Private production with 310pre6 of single photons flat pt
        
# official RelVal 310pre6 single Photons pt=35GeV
         '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0002/C8A1B6A0-1733-DE11-B770-000423D98834.root'
        
  # official RelVal 310pre6 single Photons pt=10GeV    
#    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_31X_v1/0002/CCDDC290-1733-DE11-894B-000423D991D4.root'

  # official RelVal 310pre6 RelValH130GGgluonfusion
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0002/B4C5BC38-AB32-DE11-A7BF-000423D98B5C.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0002/A67F8F78-A632-DE11-B4F1-000423D99264.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0002/8E28B621-AF32-DE11-BFE9-000423D99394.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_31X_v1/0002/488F2489-1733-DE11-AE28-001617C3B65A.root'
    
  # official RelVal 310pre6 GammaJets_Pt_80_120

#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/80FD8D8D-1733-DE11-85C3-000423D6B444.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/6AB1F928-DC32-DE11-9BB6-001617C3B5D8.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/5AD5356A-DC32-DE11-B4D1-000423D9A212.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/0231F05B-DC32-DE11-813B-001617C3B6CC.root'


    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(


#private 310pre6 single photons with flat pt=10-100 GeV



# official RelVal 310pre6 single Photons pt=35GeV
  '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/ECBC25E8-D632-DE11-B5A5-001617E30CA4.root',
  '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/466DF4FF-D632-DE11-B8F5-000423D98804.root',
  '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/204F6353-1733-DE11-B409-000423D99AA2.root'

    # official RelVal 310pre6 single Photons pt=10GeV    

#    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/4440CE7D-1733-DE11-BCCA-0016177CA778.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/AE50DEE6-E232-DE11-BD1B-000423D174FE.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/F4988DA3-E332-DE11-B652-000423D99660.root'


# official RelVal 310pre6 RelValH130GGgluonfusion
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/FE6A6821-AF32-DE11-ABA2-000423D8F63C.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/EC8312A5-AC32-DE11-BEF7-000423D952C0.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/CA2EAFC3-A932-DE11-951C-000423D98B28.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/BA17BD49-A632-DE11-9459-001617DBD230.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/AEB6AD68-A432-DE11-9187-001617E30D4A.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/A803D3CB-B132-DE11-9E98-000423D6B42C.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/A6C82112-AF32-DE11-B78D-000423D98930.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/8A5E15E9-AB32-DE11-A196-000423D6AF24.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/621194E9-A632-DE11-B9E8-000423D9997E.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/600A8E1F-A832-DE11-9F29-001617DBD224.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/4E9979FB-A432-DE11-9D2D-000423D98868.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/16D24282-1733-DE11-911D-001617DBCF90.root',
#    '/store/relval/CMSSW_3_1_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/08FD1C88-A732-DE11-8017-000423D94494.root'


# official RelVal 310pre6 GammaJets_Pt_80_120

#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/FEC12819-DC32-DE11-9D5A-000423D9A212.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/EA171432-DC32-DE11-A45D-0016177CA7A0.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/D6E3946C-DC32-DE11-8CD3-001617C3B6E8.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/D67F2D51-DC32-DE11-A427-001617DBD224.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/92D5C47B-1733-DE11-B4FA-0019DB29C614.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/8CFDD966-DC32-DE11-8FF1-000423D94534.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/42FC9933-DC32-DE11-8FF7-000423D99996.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/40454154-DC32-DE11-8F26-0016177CA7A0.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/388022FE-DC32-DE11-8768-0016177CA778.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/0E430D4F-DB32-DE11-A9CF-000423D94494.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/0CF3604C-DC32-DE11-ADA8-001617E30E2C.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/0C990C19-DC32-DE11-AD0F-001617DBD5B2.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/08802AC7-DC32-DE11-813E-000423D98DC4.root'
    
    )
 )



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5



## For single gamma fla pt =10-150
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 300
#photonValidation.etMax = 30
#photonValidation.etScale = 0.10
#photonValidation.signal = True

## For single gamma pt =10
#photonValidation.minPhoEtCut = 10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.signal = True

## For single gamma pt = 35
photonValidation.minPhoEtCut = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.signal = True

## For gam Jet and higgs
#photonValidation.minPhoEtCut = 20
#photonValidation.eMax  = 500
#photonValidation.etMax = 500
#photonValidation.signal = True
## same for all
photonValidation.convTrackMinPtCut = 1.
photonValidation.rBin = 48
photonValidation.eoverpMin = 0.
photonValidation.eoverpMax = 5.


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



