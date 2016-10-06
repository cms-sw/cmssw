import FWCore.ParameterSet.Config as cms
from Validation.SiPixelPhase1CommonV.HistogramManager_cfi import *

SiPixelPhase1TrackClustersCharge = DefaultHisto.clone(
  name = "charge",
  title = "Corrected Cluster Charge",
  range_min = 0, range_max = 100, range_nbins = 200,
  xlabel = "Charge size (in ke)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = cms.VPSet(
    Specification().groupBy("").save(),
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk").saveAll(),
    Specification(PerLadder).groupBy("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade") # per-ladder and profiles
                            .save(),
    Specification(PerLayer1D).groupBy(parent("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade")) # per-layer
                             .save(),
    Specification(PerModule).groupBy("PXBarrel|PXForward/PXLayer|PXDisk/DetId").save()
  )
)

SiPixelPhase1TrackClustersSizeX = DefaultHisto.clone(
  name = "size_x",
  title = "Cluster Size X",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "Cluster size (in pixels)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = cms.VPSet(
    Specification().groupBy("").save(),
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk").saveAll(),
    Specification(PerLadder).groupBy("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade") # per-ladder and profiles
                            .save(),
    Specification(PerLayer1D).groupBy(parent("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade")) # per-layer
                             .save(),
    Specification(PerModule).groupBy("PXBarrel|PXForward/PXLayer|PXDisk/DetId").save()
  )
)

SiPixelPhase1TrackClustersSizeY = DefaultHisto.clone(
  name = "size_y",
  title = "Cluster Size Y",
  range_min = 0, range_max = 30, range_nbins = 30,
  xlabel = "Cluster size (in pixels)",
  topFolderName = "PixelPhase1V/Clusters",
  specs = cms.VPSet(
    Specification().groupBy("").save(),
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk").saveAll(),
    Specification(PerLadder).groupBy("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade") # per-ladder and profiles
                            .save(),
    Specification(PerLayer1D).groupBy(parent("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade")) # per-layer
                             .save(),
    Specification(PerModule).groupBy("PXBarrel|PXForward/PXLayer|PXDisk/DetId").save()
  )
)

SiPixelPhase1TrackClustersConf = cms.VPSet(
  SiPixelPhase1TrackClustersCharge,
  SiPixelPhase1TrackClustersSizeX,
  SiPixelPhase1TrackClustersSizeY
)


SiPixelPhase1TrackClustersAnalyzerV = cms.EDAnalyzer("SiPixelPhase1TrackClustersV",
        clusters = cms.InputTag("siPixelClusters"),
        trajectories = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackClustersHarvesterV = cms.EDAnalyzer("SiPixelPhase1HarvesterV",
        histograms = SiPixelPhase1TrackClustersConf,
        geometry = SiPixelPhase1Geometry
)
