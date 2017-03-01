import FWCore.ParameterSet.Config as cms
process = cms.Process("photonPostprocessing")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Validation.RecoEgamma.photonPostprocessing_cfi")


process.DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)

from Validation.RecoEgamma.photonPostprocessing_cfi import *
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = "PhotonValidationRelVal310_SingleGammaPt35.root"



process.source = cms.Source("EmptySource"
)

photonPostprocessing.rBin = 48
process.p1 = cms.Path(process.photonPostprocessing)
process.schedule = cms.Schedule(process.p1)


