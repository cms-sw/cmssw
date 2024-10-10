import FWCore.ParameterSet.Config as cms
from CalibPPS.ESProducers.ctppsCompositeESSource_cfi import ctppsCompositeESSource as _esComp
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import use_single_infinite_iov_entry, p2022
from CalibPPS.ESProducers.ppsAssociationCuts_non_DB_cff import ppsAssociationCutsESSource as _esAssCuts
from Geometry.VeryForwardGeometry.commons_cff import cloneGeometry
from SimPPS.DirectSimProducer.profiles_2023_cff import profile_2023_PostTS1, profile_2023_PreTS1A, profile_2023_PreTS1B
from SimPPS.DirectSimProducer.simPPS2017_cfi import rpIds

ppsAssociationCutsESSource = _esAssCuts.clone()
use_single_infinite_iov_entry(ppsAssociationCutsESSource, p2022)
XMLIdealGeometryESSource_CTPPS, _ctppsGeometryESModule = cloneGeometry('Geometry.VeryForwardGeometry.geometryRPFromDD_2022_cfi')
# not cloning the ctppsGeometryESModule, as it is replaced by the composite ES source

ctppsCompositeESSource = _esComp.clone(
    generateEveryNEvents = 100,
    periods = [profile_2023_PostTS1,profile_2023_PreTS1A,profile_2023_PreTS1B],
    compactViewTag = _ctppsGeometryESModule.compactViewTag,
    isRun2 = _ctppsGeometryESModule.isRun2
)

