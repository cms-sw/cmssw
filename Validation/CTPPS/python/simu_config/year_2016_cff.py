import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# define era
from Configuration.Eras.Era_Run2_2016_cff import *
era = Run2_2016

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import * # using 2017 here is OK

# local reconstruction
ctppsLocalTrackLiteProducer.includeStrips = True
ctppsLocalTrackLiteProducer.includePixels = False 

reco_local = cms.Sequence(
  totemRPUVPatternFinder
  * totemRPLocalTrackFitter
  * ctppsLocalTrackLiteProducer
)

# RP ids
rpIds = cms.PSet(
  rp_45_F = cms.uint32(3),
  rp_45_N = cms.uint32(2),
  rp_56_N = cms.uint32(102),
  rp_56_F = cms.uint32(103)
)
