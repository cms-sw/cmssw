// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     CaloDetIdAssociatorMaker
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 15:05:57 GMT
//

// system include files

// user include files
#include "CaloDetIdAssociatorMaker.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CaloDetIdAssociator.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloDetIdAssociatorMaker::CaloDetIdAssociatorMaker(edm::ParameterSet const& pSet,
                                                   edm::ESConsumesCollectorT<DetIdAssociatorRecord>&& iCollector)
    : geomToken_{iCollector.consumesFrom<CaloGeometry, CaloGeometryRecord>()},
      etaBinSize{pSet.getParameter<double>("etaBinSize")},
      nPhi{pSet.getParameter<int>("nPhi")},
      nEta{pSet.getParameter<int>("nEta")} {}

std::unique_ptr<DetIdAssociator> CaloDetIdAssociatorMaker::make(const DetIdAssociatorRecord& iRecord) const {
  return make(iRecord.get(geomToken_), nPhi, nEta, etaBinSize);
}

std::unique_ptr<DetIdAssociator> CaloDetIdAssociatorMaker::make(CaloGeometry const& iGeom,
                                                                int nPhi,
                                                                int nEta,
                                                                double etaBinSize) const {
  return std::unique_ptr<DetIdAssociator>(new CaloDetIdAssociator(nPhi, nEta, etaBinSize, &iGeom));
}
