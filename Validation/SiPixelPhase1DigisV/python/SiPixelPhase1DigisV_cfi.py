import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHisto.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "ADC counts",
  range_min = 0,
  range_max = 300,
  range_nbins = 300,
  topFolderName = "PixelPhase1V/Digis",
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
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
  specs = VPSet(
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName/Event")
                            .reduce("COUNT")
                            .groupBy("PXBarrel/Shell/PXLayer/PXLadder/PXModuleName")
                            .save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName/Event")
                            .reduce("COUNT")
                            .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/PXModuleName")
                            .save(),
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
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
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
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(),
    StandardSpecification2DProfile,
 )
)

# This has to match the order of the names in the C++ enum.
SiPixelPhase1DigisConf = cms.VPSet(
  SiPixelPhase1DigisADC,
  SiPixelPhase1DigisNdigis,
  SiPixelPhase1DigisRows,
  SiPixelPhase1DigisColumns,
)

SiPixelPhase1DigisAnalyzerV = DQMStep1Module('SiPixelPhase1DigisV',
        src = cms.InputTag("simSiPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1DigisHarvesterV = DQMEDHarvester("SiPixelPhase1DigisHarvesterV",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

