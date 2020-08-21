import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2017_postTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True

#ctppsDirectProtonSimulation.empiricalAperture45="([xi]-(0.073 + [xangle] * 4.107E-04))/(39.0 + [xangle] * 0.768)"
#ctppsDirectProtonSimulation.empiricalAperture56="([xi]-(0.067 + [xangle] * 6.868E-04))/(-50.2 + [xangle] * 1.740)"

ctppsDirectProtonSimulation.empiricalAperture45="-(8.92079E-07*[xangle]-0.000150214)+(([xi]<(0.000278622*[xangle]+0.0964383))*-(3.9541e-05*[xangle]-0.0115104)+([xi]>=(0.000278622*[xangle]+0.0964383))*-(0.000108249*[xangle]-0.0249303))*([xi]-(0.000278622*[xangle]+0.0964383))"
ctppsDirectProtonSimulation.empiricalAperture56="4.56961E-05+(([xi]<(0.00075625*[xangle]+0.0643361))*-(3.01107e-05*[xangle]-0.00985126)+([xi]>=(0.00075625*[xangle]+0.0643361))*-(8.95437e-05*[xangle]-0.0169474))*([xi]-(0.00075625*[xangle]+0.0643361))"

# timing resolution
#ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2 * 0.130"
#ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2 * 0.130"

ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "2*(0.0152613+0.0498784*x-0.00824168*x^2+0.000599844*x^3-1.5923E-05*x^4)"
ctppsDirectProtonSimulation.timeResolutionDiamonds56 = "2*(-0.00458856+0.0522619*x-0.00806666*x^2+0.000558331*x^3-1.42165E-05*x^4)"
