import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2016_cff import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2016_preTS2.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# beam optics
from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import *

config_2016_preTS2 = cms.PSet(
  validityRange = cms.EventRange("0:min - 999999:max"),

  opticalFunctions = cms.VPSet(
    cms.PSet( xangle = cms.double(185), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2016_preTS2/version2/185urad.root") )
  ),

  scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(0x76100000), dirName = cms.string("XRPH_C6L5_B2"), z = cms.double(-20382.6) ),  # RP 002, strip
      cms.PSet( rpId = cms.uint32(0x76180000), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, strip
      cms.PSet( rpId = cms.uint32(0x77100000), dirName = cms.string("XRPH_C6R5_B1"), z = cms.double(+20382.6) ),  # RP 102, strip
      cms.PSet( rpId = cms.uint32(0x77180000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, strip
  )
)

ctppsOpticalFunctionsESSource.configuration.append(config_2016_preTS2)

from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True

ctppsDirectProtonSimulation.empiricalAperture45="3.76296E-05+(([xi]<0.117122)*0.00712775+([xi]>=0.117122)*0.0148651)*([xi]-0.117122)"
ctppsDirectProtonSimulation.empiricalAperture56="1.85954E-05+(([xi]<0.14324)*0.00475349+([xi]>=0.14324)*0.00629514)*([xi]-0.14324)"

# xangle/beta* options
def UseDefaultXangleBetaStar(process):
  UseCrossingAngle(185, process)

def UseDefaultXangleBetaStarDistribution(process):
  UseXangleBetaStarHistogram(process, default_xangle_beta_star_file, "2016_preTS2/h2_betaStar_vs_xangle")

# defaults
def SetDefaults(process):
  UseDefaultXangleBetaStarDistribution(process)
