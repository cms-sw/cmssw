import FWCore.ParameterSet.Config as cms

quickTrackAssociatorByHits = cms.EDProducer("QuickTrackAssociatorByHitsProducer",
	AbsoluteNumberOfHits = cms.bool(False),
	Cut_RecoToSim = cms.double(0.75),
	SimToRecoDenominator = cms.string('reco'), # either "sim" or "reco"
	Quality_SimToReco = cms.double(0.5),
	Purity_SimToReco = cms.double(0.75),
	ThreeHitTracksAreSpecial = cms.bool(True),
	associatePixel = cms.bool(True),
	associateStrip = cms.bool(True),
        useClusterTPAssociation = cms.bool(True),
        cluster2TPSrc = cms.InputTag("tpClusterProducer")
)
