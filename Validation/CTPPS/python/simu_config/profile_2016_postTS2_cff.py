import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2016_cff import *

from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2016_postTS2 as selected_optics

alignmentFile = "Validation/CTPPS/alignment/2016_postTS2.xml"

profile_2016_postTS2 = profile_base_2016.clone(
  L_int = 5.007365807,

  ctppsLHCInfo = dict(
    xangleBetaStarHistogramObject = cms.string("2016_postTS2/h2_betaStar_vs_xangle")
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = selected_optics.opticalFunctions,
    scoringPlanes = selected_optics.scoringPlanes,
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
