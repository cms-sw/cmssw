import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHisto.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "ADC counts",
  range_min = 0,
  range_max = 300,
  range_nbins = 300,
  topFolderName = "PixelPhase1V/Digis",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk/PXBModule|PXFModule").save()
  )
)

SiPixelPhase1DigisNdigis = DefaultHisto.clone(
  name = "digis", # 'Count of' added automatically
  title = "Digis",
  xlabel = "Number of Digis",
  range_min = 0,
  range_max = 30,
  range_nbins = 30,
  dimensions = 0, # this is a count
  topFolderName = "PixelPhase1V/Digis",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk/PXBModule|PXFModule").save()
  )
)

SiPixelPhase1DigisRows = DefaultHisto.clone(
  name = "row",
  title = "Digi Rows",
  xlabel = "Row",
  range_min = 0,
  range_max = 200,
  range_nbins = 200,
  topFolderName = "PixelPhase1V/Digis",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk/PXBModule|PXFModule").save()
  )
)

SiPixelPhase1DigisColumns = DefaultHisto.clone(
  name = "column",
  title = "Digi Columns",
  xlabel = "Column",
  range_min = 0,
  range_max = 300,
  range_nbins = 300,
  topFolderName = "PixelPhase1V/Digis",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/PXLayer|PXDisk/PXBModule|PXFModule").save()
  )
)

SiPixelPhase1DigisDebug = DefaultHisto.clone(
  enabled = False,
  name = "debug",
  xlabel = "Ladder #",
  range_min = 1,
  range_max = 64,
  range_nbins = 64,
  topFolderName = "PixelPhase1V/Debug",
  specs = cms.VPSet(
    Specification().groupBy("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade") 
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade"), "EXTEND_X")
                   .saveAll(),
  )
)

# This has to match the order of the names in the C++ enum.
SiPixelPhase1DigisConf = cms.VPSet(
  SiPixelPhase1DigisADC,
  SiPixelPhase1DigisNdigis,
  SiPixelPhase1DigisRows,
  SiPixelPhase1DigisColumns,
  SiPixelPhase1DigisDebug
)

SiPixelPhase1DigisAnalyzerV = cms.EDAnalyzer("SiPixelPhase1DigisV",
        src = cms.InputTag("simSiPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1DigisHarvesterV = cms.EDAnalyzer("SiPixelPhase1DigisHarvesterV",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)
