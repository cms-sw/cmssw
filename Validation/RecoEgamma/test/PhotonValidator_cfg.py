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
# input = cms.untracked.int32(1000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre3_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre3_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre3_QCD_Pt_50_80.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre3_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre3_H130GGgluonfusion.root'



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
  # official RelVal 310pre3 single Photons pt=35GeV
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_30X_v1/0001/149A7E3E-7D0A-DE11-BEF8-000423D94700.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_30X_v1/0001/1C22B073-180A-DE11-AC22-001617C3B5E4.root'

  # official RelVal 310pre3 single Photons pt=10GeV    
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_30X_v1/0001/98992FAA-7D0A-DE11-B441-001617C3B6E8.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/IDEAL_30X_v1/0001/DE9E4009-090A-DE11-A1ED-001617C3B6CC.root'

  # official RelVal 310pre3 RelValH130GGgluonfusion
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_30X_v1/0001/7A4B78E6-410A-DE11-B7C0-000423D98EA8.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_30X_v1/0001/82C64393-410A-DE11-910F-000423D94494.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_30X_v1/0001/C853A473-410A-DE11-A5BF-000423D94A20.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_30X_v1/0001/F2A0F3A2-7D0A-DE11-8D4B-001617DF785A.root'

  # official RelVal 310pre3 GammaJets_Pt_80_120
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/2C2A268C-490A-DE11-BBAE-001617E30D0A.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/2CAB972D-510A-DE11-B526-001617E30CA4.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/4AAE9505-7D0A-DE11-8311-000423D6AF24.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/C47B36A0-450A-DE11-BFF3-001617E30E2C.root'
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
  # official RelVal 310pre3 single Photons pt=35GeV
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/123EFEE3-7E0A-DE11-8173-000423D99AAA.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/12994774-180A-DE11-9B76-000423D6A6F4.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/6E9BE30D-160A-DE11-82BF-000423D33970.root'

    # official RelVal 310pre3 single Photons pt=10GeV    
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/2C4C6AA0-7E0A-DE11-81DA-000423D944DC.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/401210A6-080A-DE11-BF97-001617C3B73A.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/9C3CA9EA-080A-DE11-A8D6-001617E30F56.root'

    # official RelVal 310pre3 RelValH130GGgluonfusion
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/0C3BB57D-410A-DE11-84D2-000423D98B08.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/1A8CC1DF-410A-DE11-868B-000423D98DD4.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/2EA74839-410A-DE11-9FD3-001617C3B6FE.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/3CF0F6A1-7D0A-DE11-A715-001617DBD332.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/3E91C08F-410A-DE11-A74E-000423D98C20.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/724774B0-410A-DE11-9B35-000423D98950.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/ACC9AE14-410A-DE11-9636-000423D986C4.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/B644676D-410A-DE11-88CC-001617E30D4A.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/C4D15782-410A-DE11-B15D-000423D996C8.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DA997D0A-420A-DE11-B262-0019DB29C5FC.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DECE8989-410A-DE11-80CD-000423D996B4.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E25F5BAF-410A-DE11-8E5E-000423D99658.root',
#    '/store/relval/CMSSW_3_1_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/EC5C72F0-410A-DE11-9A60-000423D99B3E.root'

# official RelVal 310pre3 GammaJets_Pt_80_120
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/141D8F3E-800A-DE11-95ED-000423D98DD4.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/224A459C-450A-DE11-B726-000423D95220.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/303018C5-4B0A-DE11-B666-001617C3B654.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/3C52E937-480A-DE11-BDEE-0019DB29C5FC.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/56B3E48C-450A-DE11-8BBA-000423D98FBC.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/5EB75E28-420A-DE11-9454-000423D98DD4.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/9052DD97-4F0A-DE11-841D-001617E30D06.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/B2B38FAB-510A-DE11-88D8-001617E30D12.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/D21B44A4-530A-DE11-90C5-000423D8F63C.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/DEF5B628-530A-DE11-8701-000423D94534.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/E8DAF14C-460A-DE11-A97B-000423D986A8.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/FE7E15E1-420A-DE11-B9C6-000423D986A8.root',
    '/store/relval/CMSSW_3_1_0_pre3/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0001/FED13697-4F0A-DE11-AF5F-001617E30E28.root'
    
    )
 )



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5

photonValidation.minPhoEtCut = 20
photonValidation.eMax = 500
photonValidation.etMax = 500
photonValidation.convTrackMinPtCut = 1.


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
#process.p1 = cms.Path(process.mix*process.trackingParticles*process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



