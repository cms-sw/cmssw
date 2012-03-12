import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_cff import *
siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")
