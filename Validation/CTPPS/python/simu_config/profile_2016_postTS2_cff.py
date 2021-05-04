import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2016_cff import *

alignmentFile = "Validation/CTPPS/alignment/2016_postTS2.xml"

profile_2016_postTS2 = profile_base_2016.clone(
  L_int = 5.007365807,

  ctppsLHCInfo = dict(
    xangleBetaStarHistogramObject = cms.string("2016_postTS2/h2_betaStar_vs_xangle")
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = cms.VPSet(
      cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_postTS2/version2/140urad.root") )
    ),

    scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002, strip
      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102, strip
      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
    )
  ),

  ctppsRPAlignmentCorrectionsDataXML = dict(
    MisalignedFiles = [alignmentFile],
    RealFiles = [alignmentFile]
  ),

  # direct simu data
  ctppsDirectSimuData = dict(
    empiricalAperture45 = cms.string("6.10374E-05+(([xi]<0.113491)*0.00795942+([xi]>=0.113491)*0.01935)*([xi]-0.113491)"),
    empiricalAperture56 = cms.string("([xi]-0.110)/130.0")
  )
)
