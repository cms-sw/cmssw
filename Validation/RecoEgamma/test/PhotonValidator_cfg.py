import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
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
# input = cms.untracked.int32(5000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal219_SingleGammaPt35TestPCA.root'



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

 # official RelVal 3000pre2 single Photons pt=35GeV
  '/store/relval/CMSSW_3_0_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_V9_v2/0001/F0EF2B52-6FB2-DD11-84F5-0030487A18F2.root'
   
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
   # official RelVal 3000pre2 single Photons pt=35GeV 

  '/store/relval/CMSSW_3_0_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/3AB121FC-1CB4-DD11-9D43-000423D98868.root',
  '/store/relval/CMSSW_3_0_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/46D2E3E4-6EB2-DD11-B302-0030487D0D3A.root',
  '/store/relval/CMSSW_3_0_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/70F19AE4-6EB2-DD11-A3D1-0030487C608C.root'
  
    
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

process.p1 = cms.Path(process.tpSelection*process.photonValidation)
process.schedule = cms.Schedule(process.p1)



