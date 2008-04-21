import FWCore.ParameterSet.Config as cms

# needed geometries
#
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#es_module = EcalPreshowerGeometryEP {}
#es_module = EcalBarrelGeometryEP {}
#es_module = EcalEndcapGeometryEP {}
#es_module = HcalHardcodeGeometryEP {}
#es_module = CaloTowerHardcodeGeometryEP {}
#es_module = CaloGeometryBuilder {}
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
#include "Geometry/ForwardGeometry/data/ForwardGeometry.cff"
# actual producer
from Validation.GlobalHits.globalhits_prodhist_cfi import *

