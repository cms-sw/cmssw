// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     MuonDetIdAssociatorMaker
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 15:05:57 GMT
//

// system include files

// user include files
#include "MuonDetIdAssociatorMaker.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MuonDetIdAssociator.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonDetIdAssociatorMaker::MuonDetIdAssociatorMaker(edm::ParameterSet const& pSet,
                                                   edm::ESConsumesCollectorT<DetIdAssociatorRecord>&& iCollector)
    : etaBinSize{pSet.getParameter<double>("etaBinSize")},
      nPhi{pSet.getParameter<int>("nPhi")},
      nEta{pSet.getParameter<int>("nEta")},
      includeBadChambers_{pSet.getParameter<bool>("includeBadChambers")},
      includeGEM_{pSet.getParameter<bool>("includeGEM")},
      includeME0_{pSet.getParameter<bool>("includeME0")} {
  geomToken_ = iCollector.consumes();
  badChambersToken_ = iCollector.consumes();
}

std::unique_ptr<DetIdAssociator> MuonDetIdAssociatorMaker::make(const DetIdAssociatorRecord& iRecord) const {
  return std::unique_ptr<DetIdAssociator>(new MuonDetIdAssociator(nPhi,
                                                                  nEta,
                                                                  etaBinSize,
                                                                  &iRecord.get(geomToken_),
                                                                  &iRecord.get(badChambersToken_),
                                                                  includeBadChambers_,
                                                                  includeGEM_,
                                                                  includeME0_));
}
