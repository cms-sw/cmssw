import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

# define era
from Configuration.Eras.Era_Run2_2018_cff import *
era = Run2_2018

# beam optics
from CalibPPS.ESProducers.ctppsOpticalFunctionsESSource_cfi import *

config_2018 = cms.PSet(
  validityRange = cms.EventRange("0:min - 999999:max"),

  opticalFunctions = cms.VPSet(
      cms.PSet( xangle = cms.double(120), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/120urad.root") ),
      cms.PSet( xangle = cms.double(130), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/130urad.root") ),
      cms.PSet( xangle = cms.double(140), fileName = cms.FileInPath("CalibPPS/ESProducers/data/optical_functions/2018/version4/140urad.root") )
  ),

  scoringPlanes = cms.VPSet(
      # z in cm
      cms.PSet( rpId = cms.uint32(2014838784), dirName = cms.string("XRPH_D6L5_B2"), z = cms.double(-21255.1) ),  # RP 003, pixel
      cms.PSet( rpId = cms.uint32(2023227392), dirName = cms.string("XRPH_B6L5_B2"), z = cms.double(-21955.0) ),  # RP 023, pixel
      cms.PSet( rpId = cms.uint32(2031616000), dirName = cms.string("XRPH_D6R5_B1"), z = cms.double(+21255.1) ),  # RP 103, pixel
      cms.PSet( rpId = cms.uint32(2040004608), dirName = cms.string("XRPH_B6R5_B1"), z = cms.double(+21955.0) ),  # RP 123, pixel
  )
)

ctppsOpticalFunctionsESSource.configuration.append(config_2018)

from CalibPPS.ESProducers.ctppsInterpolatedOpticalFunctionsESSource_cfi import *
ctppsInterpolatedOpticalFunctionsESSource.lhcInfoLabel = ""

# geometry
from Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi import *

# alignment
from CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
alignmentFile = "Validation/CTPPS/alignment/2018.xml"
ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = [alignmentFile]
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = [alignmentFile]

# aperture cuts
ctppsDirectProtonSimulation.useEmpiricalApertures = True
ctppsDirectProtonSimulation.empiricalAperture45_xi0_int = 0.079
ctppsDirectProtonSimulation.empiricalAperture45_xi0_slp = 4.211E-04
ctppsDirectProtonSimulation.empiricalAperture45_a_int = 42.8
ctppsDirectProtonSimulation.empiricalAperture45_a_slp = 0.669
ctppsDirectProtonSimulation.empiricalAperture56_xi0_int = 0.074
ctppsDirectProtonSimulation.empiricalAperture56_xi0_slp = 6.604E-04
ctppsDirectProtonSimulation.empiricalAperture56_a_int = -22.7
ctppsDirectProtonSimulation.empiricalAperture56_a_slp = 1.600

# local reconstruction
ctppsLocalTrackLiteProducer.includeStrips = False
ctppsLocalTrackLiteProducer.includePixels = True 

reco_local = cms.Sequence(
  ctppsPixelLocalTracks
  * ctppsLocalTrackLiteProducer
)

# RP ids
rpIds = cms.PSet(
  rp_45_F = cms.uint32(23),
  rp_45_N = cms.uint32(3),
  rp_56_N = cms.uint32(103),
  rp_56_F = cms.uint32(123)
)

# defaults
def SetDefaults(process):
  UseCrossingAngle(140, process)
