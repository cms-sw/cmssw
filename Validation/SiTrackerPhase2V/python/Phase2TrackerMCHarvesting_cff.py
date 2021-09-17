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
Phase2OTRechitHarvester_PS.resXvseta.name = cms.string('resolutionXFitvseta_Pixel')
Phase2OTRechitHarvester_PS.resYvseta.name = cms.string('resolutionYFitvseta_Pixel')
Phase2OTRechitHarvester_PS.resXvsphi.name = cms.string('resolutionXFitvsphi_Pixel')
Phase2OTRechitHarvester_PS.resYvsphi.name = cms.string('resolutionYFitvsphi_Pixel')
Phase2OTRechitHarvester_PS.meanXvseta.name = cms.string('meanXFitvseta_Pixel')
Phase2OTRechitHarvester_PS.meanYvseta.name = cms.string('meanYFitvseta_Pixel')
Phase2OTRechitHarvester_PS.meanXvsphi.name = cms.string('meanXFitvsphi_Pixel')
Phase2OTRechitHarvester_PS.meanYvsphi.name = cms.string('meanYFitvsphi_Pixel')

Phase2OTRechitHarvester_2S=Phase2OTRechitHarvester_PS.clone(
    NbarrelLayers = cms.uint32(3),
    NDisk1Rings = cms.uint32(15),
    NDisk2Rings = cms.uint32(11),
    ResidualXvsEta = cms.string('Delta_X_vs_Eta_Strip'),
    ResidualXvsPhi = cms.string('Delta_X_vs_Phi_Strip'),
    ResidualYvsEta = cms.string('Delta_Y_vs_Eta_Strip'),
    ResidualYvsPhi = cms.string('Delta_Y_vs_Phi_Strip'),

)
Phase2OTRechitHarvester_2S.resXvseta.name = cms.string('resolutionXFitvseta_Strip')
Phase2OTRechitHarvester_2S.resYvseta.name = cms.string('resolutionYFitvseta_Strip')
Phase2OTRechitHarvester_2S.resXvsphi.name = cms.string('resolutionXFitvsphi_Strip')
Phase2OTRechitHarvester_2S.resYvsphi.name = cms.string('resolutionYFitvsphi_Strip')
Phase2OTRechitHarvester_2S.meanXvseta.name = cms.string('meanXFitvseta_Strip')
Phase2OTRechitHarvester_2S.meanYvseta.name = cms.string('meanYFitvseta_Strip')
Phase2OTRechitHarvester_2S.meanXvsphi.name = cms.string('meanXFitvsphi_Strip')
Phase2OTRechitHarvester_2S.meanYvsphi.name = cms.string('meanYFitvsphi_Strip')

#OTTracking rechit
Phase2OTTrackingRechitHarvester_PS=Phase2OTRechitHarvester_PS.clone(
    TopFolder = cms.string('TrackerPhase2OTTrackingRecHitV')
)

Phase2OTTrackingRechitHarvester_2S=Phase2OTRechitHarvester_2S.clone(
    TopFolder = cms.string('TrackerPhase2OTTrackingRecHitV')
)

trackerphase2ValidationHarvesting = cms.Sequence(Phase2ITRechitHarvester
                                                 * Phase2ITtrackingrechitHarvester
                                                 * Phase2OTTrackingRechitHarvester_PS
                                                 * Phase2OTTrackingRechitHarvester_2S
)

from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toReplaceWith(trackerphase2ValidationHarvesting, trackerphase2ValidationHarvesting.copyAndExclude([Phase2OTTrackingRechitHarvester_PS,Phase2OTTrackingRechitHarvester_2S]))

trackerphase2ValidationHarvesting_standalone = cms.Sequence(Phase2ITRechitHarvester
                                                            * Phase2ITtrackingrechitHarvester
                                                            * Phase2OTRechitHarvester_PS
                                                            * Phase2OTRechitHarvester_2S
                                                            * Phase2OTTrackingRechitHarvester_PS
                                                            * Phase2OTTrackingRechitHarvester_2S
)
