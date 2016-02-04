import FWCore.ParameterSet.Config as cms

# needed geometries
#
#es_module = EcalPreshowerGeometryEP {}
#es_module = EcalBarrelGeometryEP {}
#es_module = EcalEndcapGeometryEP {}
#es_module = HcalHardcodeGeometryEP {}
#es_module = CaloTowerHardcodeGeometryEP {}
#es_module = CaloGeometryBuilder {}
#include "Geometry/ForwardGeometry/data/ForwardGeometry.cff"
# actual producer
from Validation.GlobalHits.globalhits_prodhist_cfi import *

