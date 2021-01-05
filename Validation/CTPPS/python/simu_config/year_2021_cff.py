import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# base profile settings for 2021
profile_base_2021 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 6500
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = cms.VPSet(
      cms.PSet( xangle = cms.double(110.444), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/110.444urad.root") ),
      cms.PSet( xangle = cms.double(184.017), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2021/version_pre3/184.017urad.root") )
    ),

    scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, pixel
      cms.PSet( rpId = cms.uint32(2056257536), dirName = cms.string("XRPH_A6L5_B2"), z = cms.double(-21507.8) ),  # RP 022, diamond
      cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
      cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

      cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, pixel
      cms.PSet( rpId = cms.uint32(2073034752), dirName = cms.string("XRPH_A6R5_B1"), z = cms.double(+21507.8) ),  # RP 122, diamond
      cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
      cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
    )
  ),
 
  ctppsDirectSimuData = dict(
    empiricalAperture45 = cms.string("1E3*([xi] - 0.20)"),
    empiricalAperture56 = cms.string("1E3*([xi] - 0.20)")
  )
)

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi import *
ctppsCompositeESSource.compactViewTag = ctppsGeometryESModule.compactViewTag
del ctppsGeometryESModule # this functionality is replaced by the composite ES source

# local reconstruction
ctppsLocalTrackLiteProducer.includeStrips = False
ctppsLocalTrackLiteProducer.includePixels = True
ctppsLocalTrackLiteProducer.includeDiamonds = True

reco_local = cms.Sequence(
  ctppsPixelLocalTracks
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
from Validation.CTPPS.simu_config.profile_2021_default_cff import profile_2021_default
ctppsCompositeESSource.periods = [profile_2021_default]
