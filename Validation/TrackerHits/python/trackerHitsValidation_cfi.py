import FWCore.ParameterSet.Config as cms

#   module trackerHitsValid = TrackerHitProducer
trackerHitsValid = cms.EDAnalyzer("TrackerHitAnalyzer",
    G4TrkSrc = cms.InputTag("g4SimHits"),
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    Name = cms.untracked.string('TrackerHitAnalyzer'),
    Verbosity = cms.untracked.bool(False),
    runStandalone = cms.bool(False),
    outputFile =cms.untracked.string(''),
    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),

    Label = cms.string('TrkHits'),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    SiTOBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),
    SiTIBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
    SiTIBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),
    SiTOBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
    SiTECHighSrc = cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
    SiTECLowSrc = cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),
    SiTIDHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIDHighTof")
)


