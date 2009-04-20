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
photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaPt35.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaPt10.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_SingleGammaFlatPt10_100.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_QCD_Pt_50_80.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_GammaJets_Pt_80_120.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal310pre5_H130GGgluonfusion.root'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#'file:/build/nancy/CMSSW_3_1_0_pre5/src/RecoEgamma/SingleGammaPt35_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO.root'

  # official RelVal 310pre5 single Photons pt=35GeV
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_31X_v1/0000/86868101-972B-DE11-A12D-000423D98EA8.root',
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_31X_v1/0000/F43FF202-0C2C-DE11-BA8C-000423D6B358.root'

  # official RelVal 310pre5 single Photons pt=10GeV    


  # official RelVal 310pre5 RelValH130GGgluonfusion


  # official RelVal 310pre5 GammaJets_Pt_80_120

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
  # official RelVal 310pre5 single Photons pt=35GeV
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/26334D80-962B-DE11-8BC5-000423D6C8EE.root',
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/403EA350-962B-DE11-8642-000423D98800.root',
    '/store/relval/CMSSW_3_1_0_pre5/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/C087DE04-972B-DE11-836D-000423D9853C.root'

    # official RelVal 310pre5 single Photons pt=10GeV    

    # official RelVal 310pre5 RelValH130GGgluonfusion

# official RelVal 310pre5 GammaJets_Pt_80_120

    
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



