import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2016_cff import *

from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2016_preTS2 as selected_optics

alignmentFile = "Validation/CTPPS/alignment/2016_preTS2.xml"

profile_2016_preTS2 = profile_base_2016.clone(
  L_int = 6.138092276 + 3.654039035,

  ctppsLHCInfo = dict(
    xangleBetaStarHistogramObject = cms.string("2016_preTS2/h2_betaStar_vs_xangle")
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = selected_optics.opticalFunctions,
    scoringPlanes = selected_optics.scoringPlanes,
  ),

  ctppsRPAlignmentCorrectionsDataXML = dict(
    MisalignedFiles = [alignmentFile],
    RealFiles = [alignmentFile]
  ),

  ctppsDirectSimuData = dict(
    empiricalAperture45 = cms.string("3.76296E-05+(([xi]<0.117122)*0.00712775+([xi]>=0.117122)*0.0148651)*([xi]-0.117122)"),
    empiricalAperture56 = cms.string("1.85954E-05+(([xi]<0.14324)*0.00475349+([xi]>=0.14324)*0.00629514)*([xi]-0.14324)")
  )
)
