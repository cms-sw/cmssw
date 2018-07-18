import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1RecHitsInTimeEvents = DefaultHisto.clone(
  name = "in_time_bunch",
  title = "Events (in-time bunch)",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "number of in-time rechits events",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsOutTimeEvents = DefaultHisto.clone(
  name = "out_time_bunch",
  title = "Events (out-time bunch)",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "number of out-time rechit events",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)


SiPixelPhase1RecHitsNSimHits = DefaultHisto.clone(
  name = "nsimhits",
  title = "SimHits",
  range_min = 0, range_max = 100, range_nbins = 100,
  xlabel = "sim hit event number in event",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsPosX = DefaultHisto.clone(
  name = "rechit_x",
  title = "X position of RecHits",
  range_min = -2., range_max = 2., range_nbins = 80,
  xlabel = "RecHit position X dimension",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsPosY = SiPixelPhase1RecHitsPosX.clone(
  name = "rechit_y",
  title = "Y position of RecHits",
  xlabel = "RecHit position Y dimension",
  range_min = -4., range_max = 4., range_nbins = 80,
)

SiPixelPhase1RecHitsResX = DefaultHisto.clone(
  name = "res_x",
  title = "X resolution of RecHits",
  range_min = -200., range_max = 200., range_nbins = 200,
  xlabel = "RecHit resolution X dimension",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsResY = SiPixelPhase1RecHitsResX.clone(
  name = "res_y",
  title = "Y resolution of RecHits",
  xlabel = "RecHit resolution Y dimension"
)

SiPixelPhase1RecHitsErrorX = DefaultHisto.clone(
  name = "rechiterror_x",
  title = "RecHit Error in X-direction",
  range_min = 0, range_max = 0.02, range_nbins = 100,
  xlabel = "X error",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsErrorY = SiPixelPhase1RecHitsErrorX.clone(
  name = "rechiterror_y",
  title = "RecHit Error in Y-direction",
  xlabel = "Y error"
)

SiPixelPhase1RecHitsPullX = DefaultHisto.clone(
  name = "pull_x",
  title = "RecHit Pull in X-direction",
  range_min = -10., range_max = 10., range_nbins = 100,
  xlabel = "X Pull",
  dimensions = 1,
  topFolderName = "PixelPhase1V/RecHits",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
  )
)

SiPixelPhase1RecHitsPullY = SiPixelPhase1RecHitsPullX.clone(
  name = "pull_y",
  title = "RecHit Pull in Y-direction",
  xlabel = "Y Pull"
)

SiPixelPhase1RecHitsConf = cms.VPSet(
  SiPixelPhase1RecHitsInTimeEvents,
  SiPixelPhase1RecHitsOutTimeEvents,
  SiPixelPhase1RecHitsNSimHits,
  SiPixelPhase1RecHitsPosX,
  SiPixelPhase1RecHitsPosY,
  SiPixelPhase1RecHitsResX,
  SiPixelPhase1RecHitsResY,
  SiPixelPhase1RecHitsErrorX,
  SiPixelPhase1RecHitsErrorY,
  SiPixelPhase1RecHitsPullX,
  SiPixelPhase1RecHitsPullY,
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelPhase1RecHitsAnalyzerV = DQMEDAnalyzer('SiPixelPhase1RecHitsV',
        src = cms.InputTag("siPixelRecHits"),
        # Track assoc. parameters
        associatePixel = cms.bool(True),
        ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
            'g4SimHitsTrackerHitsPixelBarrelHighTof', 
            'g4SimHitsTrackerHitsPixelEndcapLowTof', 
            'g4SimHitsTrackerHitsPixelEndcapHighTof'),
        associateStrip = cms.bool(False),
        associateRecoTracks = cms.bool(False),
        pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
        stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1RecHitsHarvesterV = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)

