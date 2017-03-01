import FWCore.ParameterSet.Config as cms

quickTrackAssociatorByHits = cms.EDProducer("QuickTrackAssociatorByHitsProducer",
	AbsoluteNumberOfHits = cms.bool(False),
	Cut_RecoToSim = cms.double(0.75),
	SimToRecoDenominator = cms.string('reco'), # either "sim" or "reco"
	Quality_SimToReco = cms.double(0.5),
	Purity_SimToReco = cms.double(0.75),
	ThreeHitTracksAreSpecial = cms.bool(True),
        PixelHitWeight = cms.double(1.0),
	associatePixel = cms.bool(True),
	associateStrip = cms.bool(True),
        pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
        stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
        useClusterTPAssociation = cms.bool(True),
        cluster2TPSrc = cms.InputTag("tpClusterProducer")
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    quickTrackAssociatorByHits.associateStrip = False
    quickTrackAssociatorByHits.associatePixel = False
    quickTrackAssociatorByHits.useClusterTPAssociation = False

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( 
    quickTrackAssociatorByHits,
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis","Pixel"),
    stripSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker")
)
