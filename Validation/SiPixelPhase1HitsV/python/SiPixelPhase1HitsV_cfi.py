import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1HitsEnergyLoss = DefaultHisto.clone(
  name = "eloss",
  title = "Energy loss",
  range_min = 0, range_max = 0.001, range_nbins = 10000,
  xlabel = "Energy Loss",
  dimensions = 1,
  topFolderName = "PixelPhase1V/Hits",
  specs = VPSet(
   Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
   Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1HitsEntryExitX = DefaultHisto.clone(
  name = "entry_exit_x",
  title = "Entryx-Exitx",
  range_min = -0.03, range_max = 0.03, range_nbins = 10000,
  xlabel = "",
  dimensions = 1,
  topFolderName = "PixelPhase1V/Hits",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1HitsEntryExitY = SiPixelPhase1HitsEntryExitX.clone(
  name = "entry_exit_y",
  title = "Entryy-Exity",
  xlabel = "",
  range_min = -0.03, range_max = 0.03, range_nbins = 10000,
)

SiPixelPhase1HitsEntryExitZ = SiPixelPhase1HitsEntryExitX.clone(
  name = "entry_exit_z",
  title = "Entryz-Exitz",
  xlabel = "",
  range_min = 0.0, range_max = 0.05, range_nbins = 10000,
)

SiPixelPhase1HitsPosX = DefaultHisto.clone(
  name = "local_x",
  title = "X position of Hits",
  range_min = -3.5, range_max = 3.5, range_nbins = 10000,
  xlabel = "Hit position X dimension",
  dimensions = 1,
  topFolderName = "PixelPhase1V/Hits",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1HitsPosY = SiPixelPhase1HitsPosX.clone(
  name = "local_y",
  title = "Y position of Hits",
  xlabel = "Hit position Y dimension",
  range_min = -3.5, range_max = 3.5, range_nbins = 10000,
)

SiPixelPhase1HitsPosZ = SiPixelPhase1HitsPosX.clone(
  name = "local_z",
  title = "Z position of Hits",
  xlabel = "Hit position Z dimension",
  range_min = -0.05, range_max = 0.05, range_nbins = 500,
)

SiPixelPhase1HitsPosPhi = SiPixelPhase1HitsPosX.clone(
  name = "local_phi",
  title = "Phi position of Hits",
  xlabel = "Hit position phi dimension",
  range_min = -3.5, range_max = 3.5, range_nbins = 10000,
)

SiPixelPhase1HitsPosEta = SiPixelPhase1HitsPosX.clone(
  name = "local_eta",
  title = "Eta position of Hits",
  xlabel = "Hit position Eta dimension",
  range_min = -0.1, range_max = 0.1, range_nbins = 1000,
)

SiPixelPhase1HitsEfficiencyTrack = DefaultHistoTrack.clone(
  name = "trackefficiency",
  title = "Track Efficiency (by hits)",
  xlabel = "#valid/(#valid+#missing)",
  dimensions = 1,
  topFolderName = "PixelPhase1V/Hits",
  specs = VPSet(
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1HitsEfficiencyTrackPt = DefaultHistoTrack.clone(
  name = "trackefficiencypt",
  title = "Track Efficiency (by hits) vs Pt",
  xlabel = "pT",
  dimensions = 1,
#  range_min = -0.1, range_max = 100.0, range_nbins = 100,
  topFolderName = "PixelPhase1V/Hits",
  specs = VPSet(
    Specification().groupBy("PXBarrel")
                   .reduce("Mean")
                   .save(),
    Specification().groupBy("PXForward")
                   .reduce("Mean")
                   .save(),
  )
)

SiPixelPhase1HitsEfficiencyTrackEta = SiPixelPhase1HitsEfficiencyTrackPt.clone(
  name = "trackefficiencyeta",
  title = "Track Efficiency (by hits) vs eta",
  xlabel = "eta",
  range_min = -7.1, range_max = 7.1, range_nbins = 100,
)

SiPixelPhase1HitsConf = cms.VPSet(
  SiPixelPhase1HitsEnergyLoss,
  SiPixelPhase1HitsEntryExitX,
  SiPixelPhase1HitsEntryExitY,
  SiPixelPhase1HitsEntryExitZ,
  SiPixelPhase1HitsPosX,
  SiPixelPhase1HitsPosY,
  SiPixelPhase1HitsPosZ,
  SiPixelPhase1HitsPosPhi,
  SiPixelPhase1HitsPosEta,
#  SiPixelPhase1HitsEfficiencyHit,
  SiPixelPhase1HitsEfficiencyTrack,
  SiPixelPhase1HitsEfficiencyTrackPt,
  SiPixelPhase1HitsEfficiencyTrackEta,
)

SiPixelPhase1HitsAnalyzerV = cms.EDAnalyzer("SiPixelPhase1HitsV",
        pixBarrelLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
        pixBarrelHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
        pixForwardLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
        pixForwardHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),

	# Hit Efficiency stuff
        associateRecoTracks = cms.bool(True),
        tracksTag = cms.InputTag("generalTracks"),
	tpTag = cms.InputTag("mix","MergedTrackTruth"),
        simTracksTag = cms.InputTag("g4SimHits",""),
        trackAssociatorByHitsTag = cms.InputTag("trackAssociatorByHits"),
        associateStrip = cms.bool(True),
        associatePixel = cms.bool(True),
        ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
          'g4SimHitsTrackerHitsPixelBarrelHighTof', 
          'g4SimHitsTrackerHitsPixelEndcapLowTof', 
          'g4SimHitsTrackerHitsPixelEndcapHighTof'),

        # Track assoc. parameters
        histograms = SiPixelPhase1HitsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1HitsHarvesterV = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1HitsConf,
        geometry = SiPixelPhase1Geometry
)
