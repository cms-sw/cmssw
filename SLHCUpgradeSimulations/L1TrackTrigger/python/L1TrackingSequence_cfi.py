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

TTTracksFromPixelDigisLargerPhi = TTTracksFromPixelDigis.clone()
TTTracksFromPixelDigisLargerPhi.phiWindowSF = cms.untracked.double(2.0)   #  default is 1.0
TrackTriggerTTTracksLargerPhi = cms.Sequence(BeamSpotFromSim*TTTracksFromPixelDigisLargerPhi)

TTTrackAssociatorFromPixelDigisLargerPhi = TTTrackAssociatorFromPixelDigis.clone()
TTTrackAssociatorFromPixelDigisLargerPhi.TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPixelDigisLargerPhi", "Level1TTTracks") )
TrackTriggerAssociatorTracksLargerPhi = cms.Sequence( TTTrackAssociatorFromPixelDigisLargerPhi )

ElectronTrackingSequence = cms.Sequence( TrackTriggerTTTracksLargerPhi + TrackTriggerAssociatorTracksLargerPhi )

# ---------------------------------------------------------------------------



FullTrackingSequence = cms.Sequence( DefaultTrackingSequence + ElectronTrackingSequence )



