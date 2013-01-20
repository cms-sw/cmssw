import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/EmptyPixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

# from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import * Double check this later
# TrackerDigiGeometryESModule.applyAlignment = False

from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)

from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
mergedtruth.volumeRadius = cms.double(100.0)
mergedtruth.volumeZ = cms.double(900.0)
mergedtruth.discardOutVolume = cms.bool(True)

#from Geometry.TrackerNumberingBuilder.pixelSLHCGeometryConstants_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerSLHCGeometry_cff import *
