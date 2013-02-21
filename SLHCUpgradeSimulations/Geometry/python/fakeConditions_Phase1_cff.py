import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/EmptyPixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

# from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import * Double check this later
# TrackerDigiGeometryESModule.applyAlignment = False


from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
mergedtruth.volumeRadius = cms.double(100.0)
mergedtruth.volumeZ = cms.double(900.0)
mergedtruth.discardOutVolume = cms.bool(True)

#from Geometry.TrackerNumberingBuilder.pixelSLHCGeometryConstants_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerSLHCGeometry_cff import *

# this assumes that the beamspot is gaussian
# remove this when the right global tag is used!
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
