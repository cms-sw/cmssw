import FWCore.ParameterSet.Config as cms

#   module trackerHitsValid = TrackerHitProducer
trackerHitsValid = cms.EDAnalyzer("TrackerHitAnalyzer",
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    #      untracked string Name = "TrackerHitProducer"
    Name = cms.untracked.string('TrackerHitAnalyzer'),
    Verbosity = cms.untracked.int32(3), ## verbosity inclusive. 0 provides no output

    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
    # 1 provides basic output
    # 2 provides output of the fill step
    # 3 provides output of the store step
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


