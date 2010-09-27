import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_R39F16_smpx_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/EmptyPixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/R39F16_smpx/PixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")
