import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

CkfBaseTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),

#--- Cuts applied to completed trajectory
# At least this many hits (counting matched hits as 1)
    minimumNumberOfHits = cms.int32(5),
# add this if seed is a Pair  (opposed to a triplet)
    seedPairPenalty = cms.int32(0),
# What is this ?
    chargeSignificance = cms.double(-1.0),
    #chargeSignificance = cms.double(3.0),

#--- Cuts applied after each new hit added to trajectory
# Apply Pt cut to trajectories with at least this many hits,
# accepting tracks slightly below Pt cut if statistical error permits.
    minPt = cms.double(0.9),
    nSigmaMinPt = cms.double(5.0),
    minHitsMinPt = cms.int32(3),
# Cuts on number of hits on tracks.
    #maxLostHits = cms.int32(1),
    maxLostHits = cms.int32(999), #filter replaced by maxLostHitsFraction
    maxConsecLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(100),

# Cuts on fraction of lost hits on track
    maxLostHitsFraction = cms.double(1./10),
    constantValueForLostHitsFractionFilter = cms.double(2.),

# Cut on the length of the seed extention (no lost hits allowed)
    seedExtension = cms.int32(0),
    strictSeedExtension = cms.bool(False),

# Cuts for looperTrajectoryFilter
    minNumberOfHitsForLoopers           = cms.int32(13),
    minNumberOfHitsPerLoop              = cms.int32(4),
    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),

# Cut on CCC hits
    maxCCCLostHits = cms.int32(9999),
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)


ChargeSignificanceTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ChargeSignificanceTrajectoryFilter'),
    chargeSignificance = cms.double(-1.0)
)
CompositeTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet()
)
MaxConsecLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxConsecLostHitsTrajectoryFilter'),
    maxConsecLostHits = cms.int32(1)
)
MaxLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxLostHitsTrajectoryFilter'),
    maxLostHits = cms.int32(2)
)
MaxHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxHitsTrajectoryFilter'),
    maxNumberOfHits = cms.int32(100)
)
MinHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinHitsTrajectoryFilter'),
    minimumNumberOfHits = cms.int32(5)
)
MinPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinPtTrajectoryFilter'),
    minPt = cms.double(1.0),
    nSigmaMinPt = cms.double(5.0),
    minHitsMinPt = cms.int32(3)
)
ThresholdPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ThresholdPtTrajectoryFilter'),
    nSigmaThresholdPt = cms.double(5.0),
    minHitsThresholdPt = cms.int32(3),
    thresholdPt = cms.double(10.0)
)
MaxCCCLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxCCCLostHitsTrajectoryFilter'),
    maxCCCLostHits = cms.int32(3),
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)

