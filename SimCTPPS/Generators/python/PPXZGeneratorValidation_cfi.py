import FWCore.ParameterSet.Config as cms

ppxzGeneratorValidation = cms.EDAnalyzer("PPXZGeneratorValidation",
  verbosity = cms.untracked.uint32(0),

  tagHepMC = cms.InputTag("generator", "unsmeared"),
  tagRecoTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
  tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
  tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

  outputFile = cms.string("ppxzGeneratorValidation.root")
)
