import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisADC = DefaultHisto.clone(
  name = "adc",
  title = "Digi ADC values",
  xlabel = "ADC counts",
  range_min = 0,
  range_max = 256,
  range_nbins = 256,
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
  range_max = 180,
  range_nbins = 180,
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
  range_max = 420,
  range_nbins = 420,
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

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelPhase1DigisAnalyzerV = DQMEDAnalyzer('SiPixelPhase1DigisV',
        src = cms.InputTag("simSiPixelDigis"), 
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1DigisHarvesterV = DQMEDHarvester("SiPixelPhase1DigisHarvesterV",
        histograms = SiPixelPhase1DigisConf,
        geometry = SiPixelPhase1Geometry
)

