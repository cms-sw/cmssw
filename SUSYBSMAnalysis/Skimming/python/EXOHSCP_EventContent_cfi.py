import FWCore.ParameterSet.Config as cms
from Configuration.EventContent.EventContent_cff import *

exoticaHSCPOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("exoticaHSCPSkimPath") #the selector name must be same as the path name in EXOHSCP_cfg.py in test directory.
      ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOHSCP'), #name a name you like.
        dataTier = cms.untracked.string('EXOGroup')
    ),
    fileName = cms.untracked.string('exoticahscptest.root') # can be modified later in EXOHSCP_cfg.py in  test directory. 
)


#default output contentRECOSIMEventContent
exoticaHSCPOutputModule.outputCommands.extend(RECOSIMEventContent.outputCommands)

#add specific content you need. 
SpecifiedEvenetContent=cms.PSet(
    outputCommands = cms.untracked.vstring(
      "drop *",
#       "keep *",
#      "keep *_genParticles_*_*",
#      "keep *_met_*_*",
#      "keep *_kt6CaloJets_*_*",
      "keep recoMuons_muons_*_*",
      "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep recoTracks_generalTracksSkim_*_*",
      "keep recoTrackExtras_generalTracksSkim_*_*",
      "keep TrackingRecHitsOwned_generalTracksSkim_*_*",
#      "keep SiStripClusteredmNewDetSetVector_*_*_*",
#      "keep SiPixelClusteredmNewDetSetVector_*_*_*",
#      "keep recoTracks_generalTracks_*_*",
#      "keep recoTrackExtras_generalTracks_*_*",
#      "keep TrackingRecHitsOwned_generalTracks_*_*",
      'keep *_dt1DRecHits_*_*',
      'keep *_dt4DSegments_*_*',
      'keep *_csc2DRecHits_*_*',
      'keep *_cscSegments_*_*',
      'keep *_rpcRecHits_*_*',
      'keep recoTracks_standAloneMuons_*_*',
      'keep recoTrackExtras_standAloneMuons_*_*',
      'keep TrackingRecHitsOwned_standAloneMuons_*_*',
      'keep recoTracks_globalMuons_*_*',
      'keep recoTrackExtras_globalMuons_*_*',
      'keep TrackingRecHitsOwned_globalMuons_*_*',
#      'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEB_*_*',
#      'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEE_*_*',
      'keep HBHERecHitsSorted_reducedHSCPhbhereco__*',
#      'keep HBHERecHitsSorted_hbhereco__*',
#      "keep *_*_*_EXOHSCPSkim",
      )
    )
exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)





