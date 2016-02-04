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
      "keep GenEventInfoProduct_generator_*_*",
      "keep L1GlobalTriggerReadoutRecord_*_*_*",
      "keep recoVertexs_offlinePrimaryVertices_*_*",
      "keep recoMuons_muonsSkim_*_*",
      "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
      "keep recoTracks_generalTracksSkim_*_*",
      "keep recoTrackExtras_generalTracksSkim_*_*",
      "keep TrackingRecHitsOwned_generalTracksSkim_*_*",
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
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEB_*_*',
      'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEE_*_*',
      'keep HBHERecHitsSorted_reducedHSCPhbhereco__*',
      'keep edmTriggerResults_TriggerResults__*',
      'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_HSCPIsolation01__*',
	'keep *_HSCPIsolation03__*',
	'keep *_HSCPIsolation05__*',
        'keep recoPFJets_ak5PFJets__*', 
        'keep recoPFMETs_pfMet__*',
	  'keep recoBeamSpot_offlineBeamSpot__*',
      )
    )
exoticaHSCPOutputModule.outputCommands.extend(SpecifiedEvenetContent.outputCommands)


