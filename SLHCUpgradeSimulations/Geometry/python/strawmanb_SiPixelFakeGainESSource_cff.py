import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelESProducers.SiPixelFakeGainESSource_cfi import *

SiPixelFakeGainESSource.file = cms.FileInPath(
"SLHCUpgradeSimulations/Geometry/data/strawmanb/PixelSkimmedGeometry.txt"
)
