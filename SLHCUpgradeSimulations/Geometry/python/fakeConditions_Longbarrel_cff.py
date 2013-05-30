import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry_empty.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/longbarrel/PixelSkimmedGeometry.txt')
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
SiStripApvGainGenerator.MeanGain=cms.double(1.0)
SiStripApvGainGenerator.SigmaGain=cms.double(0.0)
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
MeasurementTracker.stripClusterProducer=cms.string('')
MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
MeasurementTracker.MaskBadAPVFibers            = cms.bool(False)
MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)

from RecoTracker.MeasurementDet.OnDemandMeasurementTrackerESProducer_cfi import *
OnDemandMeasurementTracker.stripClusterProducer=cms.string('')
OnDemandMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
OnDemandMeasurementTracker.UseStripCablingDB           = cms.bool(False)
OnDemandMeasurementTracker.UseStripNoiseDB             = cms.bool(False)

from RecoVertex.BeamSpotProducer.BeamSpotFakeParameters_cfi import *
BeamSpotFakeConditions.X0 = cms.double(0.0)
BeamSpotFakeConditions.Y0 = cms.double(0.0)
BeamSpotFakeConditions.Z0 = cms.double(0.0)
BeamSpotFakeConditions.dxdz = cms.double(0.0)
BeamSpotFakeConditions.dydz = cms.double(0.0)
BeamSpotFakeConditions.sigmaZ = cms.double(5.3)
BeamSpotFakeConditions.widthX = cms.double(0.0015)
BeamSpotFakeConditions.widthY = cms.double(0.0015)
BeamSpotFakeConditions.emittanceX = cms.double(0.)
BeamSpotFakeConditions.emittanceY = cms.double(0.)
BeamSpotFakeConditions.betaStar = cms.double(0.)
BeamSpotFakeConditions.errorX0 = cms.double(0.00002)
BeamSpotFakeConditions.errorY0 = cms.double(0.00002)
BeamSpotFakeConditions.errorZ0 = cms.double(0.04000)
BeamSpotFakeConditions.errordxdz = cms.double(0.0)
BeamSpotFakeConditions.errordydz = cms.double(0.0)
BeamSpotFakeConditions.errorSigmaZ = cms.double(0.03000)
BeamSpotFakeConditions.errorWidth = cms.double(0.00003)

es_prefer_beamspot = cms.ESPrefer("BeamSpotFakeConditions","")

from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
mergedtruth.volumeRadius = cms.double(100.0)
mergedtruth.volumeZ = cms.double(900.0)
mergedtruth.discardOutVolume = cms.bool(True)



