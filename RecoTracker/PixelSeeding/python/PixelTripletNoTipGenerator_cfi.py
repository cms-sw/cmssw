import FWCore.ParameterSet.Config as cms

PixelTripletNoTipGenerator = cms.PSet(
    maxElement = cms.uint32(10000),
  ComponentName = cms.string('PixelTripletNoTipGenerator'),
  extraHitPhiToleranceForPreFiltering = cms.double(0.3),
  extraHitRPhitolerance = cms.double(0.000),
  extraHitRZtolerance = cms.double(0.030),
  nSigma = cms.double(3.0),
  chi2Cut = cms.double(25.)
)
