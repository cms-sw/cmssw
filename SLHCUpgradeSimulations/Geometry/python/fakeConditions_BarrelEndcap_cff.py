import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/BarrelEndcap/EmptyPixelSkimmedGeometry.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/BarrelEndcap/PixelSkimmedGeometry.txt')
        )
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")


from RecoVertex.BeamSpotProducer.BeamSpotFakeParameters_cfi import *
BeamSpotFakeConditions.X0 = cms.double(0.0)
BeamSpotFakeConditions.Y0 = cms.double(0.0)
BeamSpotFakeConditions.Z0 = cms.double(0.0)
BeamSpotFakeConditions.dxdz = cms.double(0.0)
BeamSpotFakeConditions.dydz = cms.double(0.0)
BeamSpotFakeConditions.sigmaZ = cms.double(5.3)
BeamSpotFakeConditions.widthX = cms.double(0.015)
BeamSpotFakeConditions.widthY = cms.double(0.015)
BeamSpotFakeConditions.emittanceX = cms.double(0.)
BeamSpotFakeConditions.emittanceY = cms.double(0.)
BeamSpotFakeConditions.betaStar = cms.double(0.)
BeamSpotFakeConditions.errorX0 = cms.double(0.00208)
BeamSpotFakeConditions.errorY0 = cms.double(0.00208)
BeamSpotFakeConditions.errorZ0 = cms.double(0.00508)
BeamSpotFakeConditions.errordxdz = cms.double(0.0)
BeamSpotFakeConditions.errordydz = cms.double(0.0)
BeamSpotFakeConditions.errorSigmaZ = cms.double(0.060)
BeamSpotFakeConditions.errorWidth = cms.double(0.0013)

es_prefer_beamspot = cms.ESPrefer("BeamSpotFakeConditions","")

from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
mergedtruth.volumeRadius = cms.double(100.0)
mergedtruth.volumeZ = cms.double(900.0)
mergedtruth.discardOutVolume = cms.bool(True)


#from Geometry.TrackerNumberingBuilder.pixelSLHCGeometryConstants_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerSLHCGeometry_cff import *


