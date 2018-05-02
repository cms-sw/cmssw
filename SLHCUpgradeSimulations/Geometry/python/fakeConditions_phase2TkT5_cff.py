import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/EmptyPixelSkimmedGeometry.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/PixelSkimmedGeometryT5.txt')
        )
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")
