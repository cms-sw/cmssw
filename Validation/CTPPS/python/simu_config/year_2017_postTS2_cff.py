import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2017_postTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True
ctppsDirectProtonSimulation.empiricalAperture45_xi0_int = 0.073
ctppsDirectProtonSimulation.empiricalAperture45_xi0_slp = 4.107E-04
ctppsDirectProtonSimulation.empiricalAperture45_a_int = 39.0
ctppsDirectProtonSimulation.empiricalAperture45_a_slp = 0.768
ctppsDirectProtonSimulation.empiricalAperture56_xi0_int = 0.067
ctppsDirectProtonSimulation.empiricalAperture56_xi0_slp = 6.868E-04
ctppsDirectProtonSimulation.empiricalAperture56_a_int = -50.2
ctppsDirectProtonSimulation.empiricalAperture56_a_slp = 1.740
