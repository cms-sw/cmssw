import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2017_postTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True
ctppsDirectProtonSimulation.empiricalAperture45="([xi]-(0.073 + [xangle] * 4.107E-04))/(39.0 + [xangle] * 0.768)"
ctppsDirectProtonSimulation.empiricalAperture56="([xi]-(0.067 + [xangle] * 6.868E-04))/(-50.2 + [xangle] * 1.740)"

# timing resolution
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * 0.130"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * 0.130"
