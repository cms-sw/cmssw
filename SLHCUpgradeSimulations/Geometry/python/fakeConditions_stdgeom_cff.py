import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/stdgeom/PixelSkimmedGeometry_stdgeom.txt')
)
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
    file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/stdgeom/PixelSkimmedGeometry_stdgeom.txt')
)
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")

from CalibTracker.SiStripESProducers.fake.SiStripNoisesFakeESSource_cfi import *
SiStripNoisesGenerator.NoiseStripLengthSlope=cms.vdouble(51.) #dec mode
SiStripNoisesGenerator.NoiseStripLengthQuote=cms.vdouble(630.)

siStripNoisesFakeESSource  = cms.ESSource("SiStripNoisesFakeESSource")
es_prefer_fake_strip_noise = cms.ESPrefer("SiStripNoisesFakeESSource",
                                                  "siStripNoisesFakeESSource")

from CalibTracker.SiStripESProducers.fake.SiStripQualityFakeESSource_cfi import *
siStripQualityFakeESSource  = cms.ESSource("SiStripQualityFakeESSource")
es_prefer_fake_strip_quality = cms.ESPrefer("SiStripQualityFakeESSource",
                                                     "siStripQualityFakeESSource")

from CalibTracker.SiStripESProducers.fake.SiStripPedestalsFakeESSource_cfi import *
siStripPedestalsFakeESSource  = cms.ESSource("SiStripPedestalsFakeESSource")
es_prefer_fake_strip_pedestal = cms.ESPrefer("SiStripPedestalsFakeESSource",
                                                     "siStripPedestalsFakeESSource")

from CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi import *
siStripLorentzAngleFakeESSource  = cms.ESSource("SiStripLorentzAngleFakeESSource")
es_prefer_fake_strip_LA = cms.ESPrefer("SiStripLorentzAngleFakeESSource",
                                               "siStripLorentzAngleFakeESSource")

siStripLorentzAngleSimFakeESSource  = cms.ESSource("SiStripLorentzAngleSimFakeESSource")
es_prefer_fake_strip_LA_sim = cms.ESPrefer("SiStripLorentzAngleSimFakeESSource",
                                                   "siStripLorentzAngleSimFakeESSource")

from CalibTracker.SiStripESProducers.fake.SiStripApvGainFakeESSource_cfi import *
SiStripApvGainGenerator.MeanGain=cms.vdouble(1.0)
SiStripApvGainGenerator.SigmaGain=cms.vdouble(0.0)
SiStripApvGainGenerator.genMode = cms.string("default")

myStripApvGainFakeESSource = cms.ESSource("SiStripApvGainFakeESSource")
es_prefer_myStripApvGainFakeESSource  = cms.ESPrefer("SiStripApvGainFakeESSource",
                                                  "myStripApvGainFakeESSource")

myStripApvGainSimFakeESSource  = cms.ESSource("SiStripApvGainSimFakeESSource")
es_prefer_myStripApvGainSimFakeESSource = cms.ESPrefer("SiStripApvGainSimFakeESSource",
                                                               "myStripApvGainSimFakeESSource")

from CalibTracker.SiStripESProducers.fake.SiStripThresholdFakeESSource_cfi import *
siStripThresholdFakeESSource  = cms.ESSource("SiStripThresholdFakeESSource")
es_prefer_fake_strip_threshold = cms.ESPrefer("SiStripThresholdFakeESSource",
                                                     "siStripThresholdFakeESSource")

from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
TrackerDigiGeometryESModule.applyAlignment = False

from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)

from RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsSimpleGaussian_cff import *
es_prefer_beamspot = cms.ESPrefer("BeamSpotFakeConditions","")

