import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import *
del ctppsGeometryESModule

# optics
profile.ctppsOpticalFunctions.opticalFunctions = cms.VPSet(
  cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/120urad.root") ),
  cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/130urad.root") ),
  cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2017/version5tim/140urad.root") )
)

profile.ctppsOpticalFunctions.scoringPlanes = cms.VPSet(
  # z in cm
  cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
  cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
  cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

  cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
  cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
  cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
)

# geometry
profile.xmlIdealGeometry.geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + ctppsUFSDGeomXMLFiles + ctppsPixelGeomXMLFiles
profile.xmlIdealGeometry.rootNodeName = cms.string('cms:CMSE')

# alignment
profile.ctppsRPAlignmentCorrectionsDataXML.RealFiles = cms.vstring("Validation/CTPPS/alignment/2017.xml")
profile.ctppsRPAlignmentCorrectionsDataXML.MisalignedFiles = cms.vstring("Validation/CTPPS/alignment/2017.xml")

# direct simu data
profile.ctppsDirectSimuData.useEmpiricalApertures = cms.bool(True)

profile.xmlIdealGeometry.geomXMLFiles.append("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml")

from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""
ctppsInterpolatedOpticalFunctionsESSource.opticsLabel = ""

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

# load profiles
from Validation.CTPPS.simu_config.profile_2017_preTS2_cff import profile_2017_preTS2
from Validation.CTPPS.simu_config.profile_2017_postTS2_cff import profile_2017_postTS2
ctppsCompositeESSource.periods = [profile_2017_postTS2, profile_2017_preTS2]
