import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2017_preTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True
ctppsDirectProtonSimulation.empiricalAperture45="(x-(0.066 + y * 3.536E-04))/(47.7 + y * 0.447)"
ctppsDirectProtonSimulation.empiricalAperture56="(x-(0.062 + y * 5.956E-04))/(-31.9 + y * 1.323)"

# timing resolution
ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * (0.0025*(x-3) + 0.080)"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * (0.0050*(x-3) + 0.060)"
