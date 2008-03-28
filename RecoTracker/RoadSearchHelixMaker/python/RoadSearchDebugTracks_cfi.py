import FWCore.ParameterSet.Config as cms

#
# standard parameter-set entries for module
#
# RoadSeachHelixMaker
#
# located in
#
# RecoTracker/RoadSearchHelixMaker
#
# 
# sequence dependency:
#
# - RoadSearchTrackCandidates: include "RecoTracker/RoadSearchTrackCandidate/data/RoadSearchTrackCandidateMaker.cfi
#
#
# service dependency:
#
# - geometry:          include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
# - tracker geometry:  include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
# - tracker numbering: include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
#
# function:
#
# produces Debug Tracks using a simple Helix Fit from RoadSearchTrackCandidates
RoadSearchDebugTracks = cms.EDFilter("RoadSearchHelixMaker",
    # module laber of RoadSearchTrackCandidateMaker
    TrackCandidateProducer = cms.string('rsTrackCandidates')
)


