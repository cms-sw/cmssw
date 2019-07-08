// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     HcalDetIdAssociatorMaker
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 15:05:57 GMT
//

// system include files

// user include files
#include "HcalDetIdAssociatorMaker.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalDetIdAssociatorMaker::HcalDetIdAssociatorMaker(edm::ParameterSet const& pSet,
                                                   edm::ESConsumesCollectorT<DetIdAssociatorRecord>&& iCollector)
    : CaloDetIdAssociatorMaker(pSet, std::move(iCollector)), hcalReg_{pSet.getParameter<int>("hcalRegion")} {}
