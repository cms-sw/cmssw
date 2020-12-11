import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# base profile settings for 2017
profile_base_2017 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 6500
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = cms.VPSet(
      cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/120urad.root") ),
      cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/130urad.root") ),
      cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/140urad.root") )
    ),

    scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
      cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
      cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
      cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
      cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
    )
  )
)

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import totemGeomXMLFiles, ctppsDiamondGeomXMLFiles, ctppsUFSDGeomXMLFiles, ctppsPixelGeomXMLFiles
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import XMLIdealGeometryESSource_CTPPS
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import ctppsGeometryESModule as _geom
ctppsCompositeESSource.compactViewTag = _geom.compactViewTag
ctppsCompositeESSource.isRun2 = _geom.isRun2

# local reconstruction
ctppsLocalTrackLiteProducer.includeStrips = True
ctppsLocalTrackLiteProducer.includePixels = True
ctppsLocalTrackLiteProducer.includeDiamonds = True

reco_local = cms.Sequence(
  totemRPUVPatternFinder
  * totemRPLocalTrackFitter
  * ctppsPixelLocalTracks
  * ctppsDiamondLocalReconstruction
  * ctppsLocalTrackLiteProducer
)

# RP ids
rpIds = cms.PSet(
  rp_45_F = cms.uint32(23),
  rp_45_N = cms.uint32(3),
  rp_56_N = cms.uint32(103),
  rp_56_F = cms.uint32(123)
)

# default list of profiles
from Validation.CTPPS.simu_config.profile_2017_preTS2_cff import profile_2017_preTS2
from Validation.CTPPS.simu_config.profile_2017_postTS2_cff import profile_2017_postTS2
ctppsCompositeESSource.periods = [profile_2017_postTS2, profile_2017_preTS2]
