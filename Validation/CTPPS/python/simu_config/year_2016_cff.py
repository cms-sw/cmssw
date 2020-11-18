import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# FIXME: move to the right place
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import * # using 2017 here is OK
del ctppsGeometryESModule

# base profile settings for 2016
profile_base_2016 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 6500
  ),

  xmlIdealGeometry = dict(
    geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + ctppsUFSDGeomXMLFiles + ctppsPixelGeomXMLFiles +
      cms.vstring("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml"),
    rootNodeName = cms.string('cms:CMSE')
  )
)

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
