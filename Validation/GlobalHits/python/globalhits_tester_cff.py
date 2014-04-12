# The following comments couldn't be translated into the new config version:

# needed geometries
#
#include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
#es_module = EcalPreshowerGeometryEP {}
#es_module = EcalBarrelGeometryEP {}
#es_module = EcalEndcapGeometryEP {}
#es_module = HcalHardcodeGeometryEP {}
#es_module = CaloTowerHardcodeGeometryEP {}
#include "Geometry/CaloEventSetup/data/CaloGeometry.cff"
#include "Geometry/CSCGeometry/data/cscGeometry.cfi"
#include "Geometry/DTGeometry/data/dtGeometry.cfi"
#include "Geometry/RPCGeometry/data/rpcGeometry.cfi"
# needed backend

import FWCore.ParameterSet.Config as cms

# actual producer
from Validation.GlobalHits.globalhits_tester_cfi import *
DQMStore = cms.Service("DQMStore")


