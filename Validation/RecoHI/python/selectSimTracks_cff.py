import FWCore.ParameterSet.Config as cms

findableSimTracks = cms.EDFilter("HitPixelLayersTPSelection",
    src = cms.InputTag("mix","MergedTrackTruth"),
	tripletSeedOnly = cms.bool(True),
	chargedOnly = cms.bool(True),
	signalOnly = cms.bool(False),
        primaryOnly = cms.bool(True),
        tpStatusBased = cms.bool(True), # for primary particle definition
	ptMin = cms.double(2.0),
	minHit = cms.int32(8),
	minRapidity = cms.double(-2.5),
	maxRapidity = cms.double(2.5),
	tip = cms.double(3.5),
	lip = cms.double(30.0),
	pdgId = cms.vint32()
)


primaryChgSimTracks = cms.EDFilter("HitPixelLayersTPSelection",
    src = cms.InputTag("mix","MergedTrackTruth"),
          tripletSeedOnly = cms.bool(False),
          chargedOnly = cms.bool(True),
          signalOnly = cms.bool(False),
          primaryOnly = cms.bool(True),
          tpStatusBased = cms.bool(True),
          ptMin = cms.double(0.1),
          minHit = cms.int32(3),
          minRapidity = cms.double(-2.5),
          maxRapidity = cms.double(2.5),
          tip = cms.double(3.5),
          lip = cms.double(30.0),
          pdgId = cms.vint32()
)


