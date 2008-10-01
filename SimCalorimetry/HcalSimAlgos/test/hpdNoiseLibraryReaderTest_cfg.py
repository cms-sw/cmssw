import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("HPDNoiseLibraryReaderTest",
      HPDNoiseLibrary = cms.PSet(
         FileName = cms.FileInPath("SimCalorimetry/HcalSimAlgos/data/hpdNoiseLibrary.root"),
	 HPDName = cms.untracked.string("HPD")
      ),
      UseBiasedHPDNoise = cms.untracked.bool(True)
    )
process.p = cms.Path(process.test)
