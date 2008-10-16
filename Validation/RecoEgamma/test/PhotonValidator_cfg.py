import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
 input = cms.untracked.int32(5000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal219_SingleGammaPt35TestPCA.root'



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

# official RelVal 219 single Photons pt=10GeV
#    '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/1A55E486-B185-DD11-B52A-000423D6A6F4.root',
#    '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/6EC604C6-AF85-DD11-9B52-000423D6B2D8.root',
#    '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/84D42DD1-B185-DD11-8A64-000423D6B5C4.root',
#    '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/B01D3DDF-0487-DD11-A5A3-001617C3B69C.root'
    
# official RelVal 219 single Photons pt=35GeV
'/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/3652B809-B585-DD11-A8D9-000423D9939C.root',
'/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/48AFC4AC-B485-DD11-A63C-000423D94C68.root',
'/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/4E76DF7F-B385-DD11-955C-000423D6A6F4.root',
'/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/725453E4-0487-DD11-A22C-000423D94494.root'




)

)



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



