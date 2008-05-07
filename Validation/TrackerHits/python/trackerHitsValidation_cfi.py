import FWCore.ParameterSet.Config as cms

#   module trackerHitsValid = TrackerHitProducer
trackerHitsValid = cms.EDAnalyzer("TrackerHitAnalyzer",
    SiTIDLowSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTIDLowTof"),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsPixelBarrelLowTof"),
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
    SiTOBLowSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTOBLowTof"),
    SiTIBHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTIBHighTof"),
    SiTIBLowSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTIBLowTof"),
    SiTOBHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTOBHighTof"),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsPixelEndcapHighTof"),
    SiTECHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTECHighTof"),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsPixelBarrelHighTof"),
    SiTECLowSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTECLowTof"),
    SiTIDHighSrc = cms.InputTag("g4SimHits","g4SimHitsTrackerHitsTIDHighTof")
)


