import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackClustersCharge = DefaultHisto.clone(
  name = "charge",
  title = "Corrected Cluster Charge",
  range_min = 0, range_max = 100, range_nbins = 200,
  xlabel = "Charge size (in ke)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1TrackClustersSizeX = DefaultHisto.clone(
  name = "size_x",
  title = "Cluster Size X",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "Cluster size (in pixels)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1TrackClustersSizeY = DefaultHisto.clone(
  name = "size_y",
  title = "Cluster Size Y",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "Cluster size (in pixels)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXDisk/PXModuleName").save(),
  )
)

SiPixelPhase1TrackClustersConf = cms.VPSet(
  SiPixelPhase1TrackClustersCharge,
  SiPixelPhase1TrackClustersSizeX,
  SiPixelPhase1TrackClustersSizeY
)


SiPixelPhase1TrackClustersAnalyzerV = cms.EDAnalyzer("SiPixelPhase1TrackClustersV",
        clusters = cms.InputTag("siPixelClusters"),
        tracks = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackClustersHarvesterV = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)
