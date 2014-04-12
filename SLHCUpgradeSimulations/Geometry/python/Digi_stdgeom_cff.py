import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Digi_cff import *
simSiPixelDigis.MissCalibrate = False
simSiPixelDigis.LorentzAngle_DB = False
simSiPixelDigis.killModules = False
simSiPixelDigis.useDB = False
simSiPixelDigis.DeadModules_DB = False

# Set AddPixelInefficiency = -20 to read in custom efficiencies
#simSiPixelDigis.AddPixelInefficiency = -20
simSiPixelDigis.thePixelColEfficiency_BPix1 = cms.double(1.-0.16)
simSiPixelDigis.thePixelColEfficiency_BPix2 = cms.double(1.-0.058)
simSiPixelDigis.thePixelColEfficiency_BPix3 = cms.double(1.-0.03)
simSiPixelDigis.thePixelColEfficiency_FPix1 = cms.double(1.-0.03)
simSiPixelDigis.thePixelColEfficiency_FPix2 = cms.double(1.-0.03)
