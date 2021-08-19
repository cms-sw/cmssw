import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.base_cff import *
import CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff as ac
ac.use_single_infinite_iov_entry(ac.ppsAssociationCutsESSource,ac.p2017)

from CalibPPS.ESProducers.ctppsOpticalFunctions_non_DB_cff import optics_2017 as selected_optics

# base profile settings for 2017
profile_base_2017 = profile_base.clone(
  ctppsLHCInfo = dict(
    beamEnergy = 6500
  ),

  ctppsOpticalFunctions = dict(
    opticalFunctions = selected_optics.opticalFunctions,
    scoringPlanes = selected_optics.scoringPlanes,
  )
)

# geometry
from Geometry.VeryForwardGeometry.commons_cff import cloneGeometry
XMLIdealGeometryESSource_CTPPS, _ctppsGeometryESModule = cloneGeometry('Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi')
ctppsCompositeESSource.compactViewTag = _ctppsGeometryESModule.compactViewTag
ctppsCompositeESSource.isRun2 = _ctppsGeometryESModule.isRun2

# local reconstruction
ctppsLocalTrackLiteProducer.includeStrips = True
ctppsLocalTrackLiteProducer.includePixels = True
ctppsLocalTrackLiteProducer.includeDiamonds = True

reco_local = cms.Sequence(
  totemRPUVPatternFinder
  * totemRPLocalTrackFitter
  * ctppsPixelLocalTracks
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
from Validation.CTPPS.simu_config.profile_2017_preTS2_cff import profile_2017_preTS2
from Validation.CTPPS.simu_config.profile_2017_postTS2_cff import profile_2017_postTS2
ctppsCompositeESSource.periods = [profile_2017_postTS2, profile_2017_preTS2]
