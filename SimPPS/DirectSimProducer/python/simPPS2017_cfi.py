import FWCore.ParameterSet.Config as cms
from CalibPPS.ESProducers.ctppsCompositeESSource_cfi import ctppsCompositeESSource as _esComp
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import use_single_infinite_iov_entry, p2017
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import ppsAssociationCutsESSource as _esAssCuts
from Geometry.VeryForwardGeometry.commons_cff import cloneGeometry
from SimPPS.DirectSimProducer.profiles_2017_cff import profile_2017_preTS2, profile_2017_postTS2

ppsAssociationCutsESSource = _esAssCuts.clone()
use_single_infinite_iov_entry(ppsAssociationCutsESSource, p2017)
XMLIdealGeometryESSource_CTPPS, ctppsGeometryESModule = cloneGeometry('Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi')

ctppsCompositeESSource = _esComp.clone(
    generateEveryNEvents = 100,
    periods = [profile_2017_preTS2, profile_2017_postTS2],
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
