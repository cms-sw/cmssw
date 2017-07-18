import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1HitsTofEta = DefaultHisto.clone(
  name = "tof_eta",
  title = "Time of flight vs #eta",
  range_min = -4.0, range_max = 4.0, range_nbins = 5000,
  range_y_min = -100, range_y_max = 100, range_y_nbins = 200,
  xlabel = "#eta", ylabel = "Time of flight",
  topFolderName = "PixelPhase1V/Hits",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1HitsTofPhi = SiPixelPhase1HitsTofEta.clone(
  name = "tof_phi",
  title = "Time of flight vs #phi",
  range_min = -3.5, range_max = 3.5, range_nbins = 5000,
  xlabel = "#phi",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("").save(),
  )
)  

SiPixelPhase1HitsTofR = SiPixelPhase1HitsTofEta.clone(
  name = "tof_r",
  title = "Time of flight vs r",
  range_min = 0, range_max = 60, range_nbins = 5000,
  xlabel = "r",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1HitsTofZ = SiPixelPhase1HitsTofEta.clone(
  name = "tof_z",
  title = "Time of flight vs z",
  range_min = -60, range_max = 60, range_nbins = 5000,
  xlabel = "z",
  dimensions = 2,
  specs = VPSet(
    Specification().groupBy("").save(),
  )
)

SiPixelPhase1HitsEnergyLoss = DefaultHisto.clone(
  name = "eloss",
  title = "Energy loss",
  range_min = 0, range_max = 0.001, range_nbins = 100,
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
  range_min = -0.03, range_max = 0.03, range_nbins = 100,
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
  range_min = -0.03, range_max = 0.03, range_nbins = 100,
)

SiPixelPhase1HitsEntryExitZ = SiPixelPhase1HitsEntryExitX.clone(
  name = "entry_exit_z",
  title = "Entryz-Exitz",
  xlabel = "",
  range_min = 0.0, range_max = 0.05, range_nbins = 100,
)

SiPixelPhase1HitsPosX = DefaultHisto.clone(
  name = "local_x",
  title = "X position of Hits",
  range_min = -3.5, range_max = 3.5, range_nbins = 100,
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
  range_min = -3.5, range_max = 3.5, range_nbins = 100,
)

SiPixelPhase1HitsPosZ = SiPixelPhase1HitsPosX.clone(
  name = "local_z",
  title = "Z position of Hits",
  xlabel = "Hit position Z dimension",
  range_min = -0.05, range_max = 0.05, range_nbins = 100,
)

SiPixelPhase1HitsPosPhi = SiPixelPhase1HitsPosX.clone(
  name = "local_phi",
  title = "Phi position of Hits",
  xlabel = "Hit position phi dimension",
  range_min = -3.5, range_max = 3.5, range_nbins = 100,
)

SiPixelPhase1HitsPosEta = SiPixelPhase1HitsPosX.clone(
  name = "local_eta",
  title = "Eta position of Hits",
  xlabel = "Hit position Eta dimension",
  range_min = -0.1, range_max = 0.1, range_nbins = 100,
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

SiPixelPhase1HitsConf = cms.VPSet(
  SiPixelPhase1HitsTofEta,
  SiPixelPhase1HitsTofPhi,
  SiPixelPhase1HitsTofR,
  SiPixelPhase1HitsTofZ,
  SiPixelPhase1HitsEnergyLoss,
  SiPixelPhase1HitsEntryExitX,
  SiPixelPhase1HitsEntryExitY,
  SiPixelPhase1HitsEntryExitZ,
  SiPixelPhase1HitsPosX,
  SiPixelPhase1HitsPosY,
  SiPixelPhase1HitsPosZ,
  SiPixelPhase1HitsPosPhi,
  SiPixelPhase1HitsPosEta,
  SiPixelPhase1HitsEfficiencyTrack,
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

SiPixelPhase1HitsHarvesterV = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1HitsConf,
        geometry = SiPixelPhase1Geometry
)
