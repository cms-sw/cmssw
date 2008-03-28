# The following comments couldn't be translated into the new config version:

# Primary trajectories producer

# Nuclear seeds producer

# Secondary tracks producer

import FWCore.ParameterSet.Config as cms

nuclearInteractionMaker = cms.EDProducer("NuclearInteractionEDProducer",
    seedsProducer = cms.string('nuclearSeed'),
    # chi2 cut on secondary tracks
    chi2Cut = cms.double(100.0),
    # Minimum distance between primary and secondary tracks
    # all secondary tracks with distance > minDistFromPrimary
    # are rejected
    minDistFromPrimary = cms.double(100.0),
    secondaryProducer = cms.string('nuclearWithMaterialTracks'),
    primaryProducer = cms.string('TrackRefitter')
)


