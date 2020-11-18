import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi import *
del ctppsGeometryESModule

# optics
profile.ctppsOpticalFunctions.opticalFunctions = cms.VPSet(
  cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/120urad.root") ),
  cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/130urad.root") ),
  cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version6/140urad.root") )
)

profile.ctppsOpticalFunctions.scoringPlanes = cms.VPSet(
  # z in cm
  cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, pixel
  cms.PSet( rpId = cms.uint32(2054160384), dirName = cms.string("XRPH_E6L5_B2"), z = cms.double(-21570.0) ),  # RP 016, diamond
  cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel

  cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, pixel
  cms.PSet( rpId = cms.uint32(2070937600), dirName = cms.string("XRPH_E6R5_B1"), z = cms.double(+21570.0) ),  # RP 116, diamond
  cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
)
 
# geometry
profile.xmlIdealGeometry.geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + ctppsUFSDGeomXMLFiles + totemTimingGeomXMLFiles + ctppsPixelGeomXMLFiles
profile.xmlIdealGeometry.rootNodeName = cms.string('cms:CMSE')

# alignment
profile.ctppsRPAlignmentCorrectionsDataXML.RealFiles = cms.vstring("Validation/CTPPS/alignment/2018.xml")
profile.ctppsRPAlignmentCorrectionsDataXML.MisalignedFiles = cms.vstring("Validation/CTPPS/alignment/2018.xml")

# direct simu data
profile.ctppsDirectSimuData.useEmpiricalApertures = cms.bool(True)
profile.ctppsDirectSimuData.empiricalAperture45 = cms.string("-(8.44219E-07*[xangle]-0.000100957)+(([xi]<(0.000247185*[xangle]+0.101599))*-(1.40289E-05*[xangle]-0.00727237)+([xi]> = (0.000247185*[xangle]+0.101599))*-(0.000107811*[xangle]-0.0261867))*([xi]-(0.000247185*[xangle]+0.101599))")
profile.ctppsDirectSimuData.empiricalAperture56 = cms.string("-(-4.74758E-07*[xangle]+3.0881E-05)+(([xi]<(0.000727859*[xangle]+0.0722653))*-(2.43968E-05*[xangle]-0.0085461)+([xi]> = (0.000727859*[xangle]+0.0722653))*-(7.19216E-05*[xangle]-0.0148267))*([xi]-(0.000727859*[xangle]+0.0722653))")

profile.xmlIdealGeometry.geomXMLFiles.append("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml")

from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""

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

# load profiles
from Validation.CTPPS.simu_config.profile_2018_preTS1_cff import profile_2018_preTS1
from Validation.CTPPS.simu_config.profile_2018_postTS2_cff import profile_2018_postTS2
from Validation.CTPPS.simu_config.profile_2018_TS1_TS2_cff import profile_2018_TS1_TS2
ctppsCompositeESSource.periods = [profile_2018_postTS2, profile_2018_preTS1, profile_2018_TS1_TS2]
