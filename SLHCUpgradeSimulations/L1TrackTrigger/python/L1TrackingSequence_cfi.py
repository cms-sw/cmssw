import FWCore.ParameterSet.Config as cms


from Configuration.StandardSequences.MagneticField_38T_PostLS1_cff import *
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *
from Configuration.Geometry.GeometryExtended2023TTIReco_cff import *
from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *

BeamSpotFromSim =cms.EDProducer("BeamSpotFromSimProducer")


# ---------------------------------------------------------------------------

#
# --- Rerun the L1Tracking, standard configuration
#     Allows to benefit from most recent improvements

from Configuration.StandardSequences.L1TrackTrigger_cff import *

DefaultTrackingSequence = cms.Sequence( TrackTriggerTTTracks + TrackTriggerAssociatorTracks)

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

#
# --- Rerun the L1Tracking, but now in the "electron configuration"
#

TTTracksFromPhase2TrackerDigisLargerPhi = TTTracksFromPhase2TrackerDigis.clone()
TTTracksFromPhase2TrackerDigisLargerPhi.phiWindowSF = cms.untracked.double(2.0)   #  default is 1.0
TrackTriggerTTTracksLargerPhi = cms.Sequence(BeamSpotFromSim*TTTracksFromPhase2TrackerDigisLargerPhi)

TTTrackAssociatorFromPhase2TrackerDigisLargerPhi = TTTrackAssociatorFromPhase2TrackerDigis.clone()
TTTrackAssociatorFromPhase2TrackerDigisLargerPhi.TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPhase2TrackerDigisLargerPhi", "Level1TTTracks") )
TrackTriggerAssociatorTracksLargerPhi = cms.Sequence( TTTrackAssociatorFromPhase2TrackerDigisLargerPhi )

ElectronTrackingSequence = cms.Sequence( TrackTriggerTTTracksLargerPhi + TrackTriggerAssociatorTracksLargerPhi )

# ---------------------------------------------------------------------------



FullTrackingSequence = cms.Sequence( DefaultTrackingSequence + ElectronTrackingSequence )



