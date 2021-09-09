import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *

from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2022 as selected_optics

# base profile settings for 2022
profile_base_2022 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 7000
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = selected_optics.opticalFunctions,
    scoringPlanes = selected_optics.scoringPlanes,
  ),
 
  ctppsDirectSimuData = dict(
    empiricalAperture45 = cms.string("1E3*([xi] - 0.20)"),
    empiricalAperture56 = cms.string("1E3*([xi] - 0.20)")
  )
)

# adjust basic settings
generator.energy = profile_base_2022.ctppsLHCInfo.beamEnergy

# geometry (NB: until a dedicated 2022 geometry is issued, it is OK to use 2021 one here)
from Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi import *
ctppsCompositeESSource.compactViewTag = ctppsGeometryESModule.compactViewTag
del ctppsGeometryESModule # this functionality is replaced by the composite ES source

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

# default list of profiles
from Validation.CTPPS.simu_config.profile_2022_default_cff import profile_2022_default
ctppsCompositeESSource.periods = [profile_2022_default]
