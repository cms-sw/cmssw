import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Validation.SiTrackerPhase2V.Phase2ITRechitHarvester_cfi import *

#ITTracking rechit
#clone the rechit harvester for tracking rechit
Phase2ITtrackingrechitHarvester=Phase2ITRechitHarvester.clone(
    TopFolder = cms.string('TrackerPhase2ITTrackingRecHitV')
)


##As of now this is to be used in standalone mode
Phase2OTRechitHarvester_PS=Phase2ITRechitHarvester.clone(
    TopFolder = cms.string('TrackerPhase2OTRecHitV'),
    NbarrelLayers = cms.uint32(3),
    NDisk1Rings = cms.uint32(10),
    NDisk2Rings = cms.uint32(7),
    EcapDisk1Name = cms.string('TEDD_1'),
    EcapDisk2Name = cms.string('TEDD_2'),
    ResidualXvsEta = cms.string('Delta_X_vs_Eta_Pixel'),
    ResidualXvsPhi = cms.string('Delta_X_vs_Phi_Pixel'),
    ResidualYvsEta = cms.string('Delta_Y_vs_Eta_Pixel'),
    ResidualYvsPhi = cms.string('Delta_Y_vs_Phi_Pixel'),
)

Phase2OTRechitHarvester_2S=Phase2OTRechitHarvester_PS.clone(
    NbarrelLayers = cms.uint32(3),
    NDisk1Rings = cms.uint32(15),
    NDisk2Rings = cms.uint32(11),
    ResidualXvsEta = cms.string('Delta_X_vs_Eta_Strip'),
    ResidualXvsPhi = cms.string('Delta_X_vs_Phi_Strip'),
    ResidualYvsEta = cms.string('Delta_Y_vs_Eta_Strip'),
    ResidualYvsPhi = cms.string('Delta_Y_vs_Phi_Strip'),

)
#OTTracking rechit
Phase2OTTrackingRechitHarvester_PS=Phase2OTRechitHarvester_PS.clone(
    TopFolder = cms.string('TrackerPhase2OTTrackingRecHitV')
)

Phase2OTTrackingRechitHarvester_2S=Phase2OTRechitHarvester_2S.clone(
    TopFolder = cms.string('TrackerPhase2OTTrackingRecHitV')
)

trackerphase2ValidationHarvesting = cms.Sequence(Phase2ITRechitHarvester*Phase2ITtrackingrechitHarvester)

trackerphase2ValidationHarvesting_standalone = cms.Sequence(Phase2ITRechitHarvester
                                                            * Phase2ITtrackingrechitHarvester
                                                            * Phase2OTRechitHarvester_PS
                                                            * Phase2OTRechitHarvester_2S
                                                            * Phase2OTTrackingRechitHarvester_PS
                                                            * Phase2OTTrackingRechitHarvester_2S
)
