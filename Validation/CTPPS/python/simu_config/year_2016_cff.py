import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# base profile settings for 2016
profile_base_2016 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 6500
  )
)

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import * # using 2017 here is OK
ctppsCompositeESSource.compactViewTag = ctppsGeometryESModule.compactViewTag
del ctppsGeometryESModule # this functionality is replaced by the composite ES source

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

# default list of profiles
from Validation.CTPPS.simu_config.profile_2016_preTS2_cff import profile_2016_preTS2
from Validation.CTPPS.simu_config.profile_2016_postTS2_cff import profile_2016_postTS2
ctppsCompositeESSource.periods = [profile_2016_postTS2, profile_2016_preTS2]
