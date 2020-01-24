import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Validation.MuonHits.muonSimHitMatcherPSet import muonSimHitMatcherPSet
from Validation.MuonGEMDigis.muonGEMDigiPSet import muonGEMDigiPSet
from Validation.MuonGEMHits.MuonGEMCommonParameters_cfi import GEMValidationCommonParameters

gemStripValidation = DQMEDAnalyzer('GEMStripDigiValidation',
  GEMValidationCommonParameters,
  folder = cms.string("MuonGEMDigisV/GEMDigisTask/Strip"),
  logCategory = cms.string("GEMStripDigiValidation"),
  detailPlot = cms.bool(True),
  gemStripDigi = muonGEMDigiPSet.gemUnpackedStripDigi,
  simSimHit = muonSimHitMatcherPSet.gemSimHit,
)
gemPadValidation = DQMEDAnalyzer('GEMPadDigiValidation',
  GEMValidationCommonParameters,
  folder = cms.string("MuonGEMDigisV/GEMDigisTask/Pad"),
  logCategory = cms.string("GEMPadDigiValidation"),
  detailPlot = cms.bool(True),
  gemPadDigi = muonGEMDigiPSet.gemPadDigi,
  simSimHit = muonSimHitMatcherPSet.gemSimHit,
)
gemClusterValidation = DQMEDAnalyzer('GEMPadDigiClusterValidation',
  GEMValidationCommonParameters,
  folder = cms.string("MuonGEMDigisV/GEMDigisTask/Cluster"),
  logCategory = cms.string("GEMPadDigiClusterValidation"),
  detailPlot = cms.bool(True),
  gemPadCluster = muonGEMDigiPSet.gemPadCluster,
  simSimHit = muonSimHitMatcherPSet.gemSimHit,
)
gemCoPadValidation = DQMEDAnalyzer('GEMCoPadDigiValidation',
  GEMValidationCommonParameters,
  folder = cms.string("MuonGEMDigisV/GEMDigisTask/CoPad"),
  logCategory = cms.string("GEMCoPadDigiValidation"),
  detailPlot = cms.bool(True),
  gemCoPadDigi = muonGEMDigiPSet.gemCoPadDigi,
  simSimHit = muonSimHitMatcherPSet.gemSimHit,
)

gemGeometryChecker = DQMEDAnalyzer('GEMCheckGeometry',
  detailPlot = cms.bool(False),
)

gemDigiValidation = cms.Sequence( gemStripValidation+
                                  gemPadValidation+
                                  gemClusterValidation+
                                  gemCoPadValidation+
                                  gemGeometryChecker)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(gemDigiValidation, gemDigiValidation.copyAndExclude([gemPadValidation,gemClusterValidation,gemCoPadValidation]))
