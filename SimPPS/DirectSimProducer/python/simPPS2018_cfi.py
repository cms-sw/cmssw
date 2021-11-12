import FWCore.ParameterSet.Config as cms
from CalibPPS.ESProducers.ctppsCompositeESSource_cfi import ctppsCompositeESSource as _esComp
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import use_single_infinite_iov_entry, p2018
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import ppsAssociationCutsESSource as _esAssCuts
from Geometry.VeryForwardGeometry.commons_cff import cloneGeometry
from SimPPS.DirectSimProducer.profiles_2018_cff import profile_2018_preTS1, profile_2018_TS1_TS2, profile_2018_postTS2

ppsAssociationCutsESSource = _esAssCuts.clone()
use_single_infinite_iov_entry(ppsAssociationCutsESSource, p2018)
XMLIdealGeometryESSource_CTPPS, ctppsGeometryESModule = cloneGeometry('Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi')

ctppsCompositeESSource = _esComp.clone(
    generateEveryNEvents = 100,
    periods = [profile_2018_preTS1, profile_2018_TS1_TS2, profile_2018_postTS2],
    compactViewTag = ctppsGeometryESModule.compactViewTag,
    isRun2 = ctppsGeometryESModule.isRun2
)

# RP ids
rpIds = cms.PSet(
  rp_45_F = cms.uint32(23),
  rp_45_N = cms.uint32(3),
  rp_56_N = cms.uint32(103),
  rp_56_F = cms.uint32(123)
)
