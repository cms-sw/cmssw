import FWCore.ParameterSet.Config as cms

RemapDetIdService = cms.Service( "RemapDetIdService",
	mapFilename=cms.FileInPath("SimTracker/SiPixelDigitizer/test/trackerDetIdMapSLHC12toSLHC13.txt"),
	inputCollections=cms.VInputTag(
		cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
		cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof")
	),
	# Note that the CMSSW version the collections above were created with just
	# has to contain one of these strings. So if e.g. "_patch3" is on the end
	# it will still remap the collection.
	versionsToRemap=cms.vstring( "CMSSW_6_2_0_SLHC11", "CMSSW_6_2_0_SLHC12" )
)
