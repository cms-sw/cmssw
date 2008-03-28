import FWCore.ParameterSet.Config as cms

#
# ctf tracks parameter-set entries for module
#
# RoadSeachTrackListCleaner
#
# located in
#
# RecoTracker/RoadSearchHelixMaker
#
# 
# sequence dependency:
#
# - ctfWithMaterialTracks: include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
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
# cleans rs Track list and put new list back in Event
ctfTrackListCleaner = cms.EDFilter("RoadSearchTrackListCleaner",
    # minimum number of RecHits used in fit
    MinFound = cms.int32(8),
    # maximum chisq/dof
    MaxNormalizedChisq = cms.double(20.0),
    # minimum pT in GeV/c
    MinPT = cms.double(0.05),
    # module laber of CTF Tracks from KF with material propagator
    TrackProducer = cms.string('ctfWithMaterialTracks')
)


