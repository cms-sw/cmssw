import FWCore.ParameterSet.Config as cms

# Define arbitrary tracker material groups
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from SimTracker.TrackerMaterialAnalysis.trackingMaterialGroups_cff import *

# Analyze and plot the tracking material
from SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_cfi import *
