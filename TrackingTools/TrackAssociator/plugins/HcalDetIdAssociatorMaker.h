#ifndef TrackingTools_TrackAssociator_HcalDetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_HcalDetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     HcalDetIdAssociatorMaker
//
/**\class HcalDetIdAssociatorMaker HcalDetIdAssociatorMaker.h "HcalDetIdAssociatorMaker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 16:11:48 GMT
//

// system include files

// user include files

// forward declarations
#include "CaloDetIdAssociatorMaker.h"
#include "HcalDetIdAssociator.h"

class HcalDetIdAssociatorMaker : public CaloDetIdAssociatorMaker {
public:
  HcalDetIdAssociatorMaker(edm::ParameterSet const&, edm::ESConsumesCollectorT<DetIdAssociatorRecord>&&);

private:
  std::unique_ptr<DetIdAssociator> make(CaloGeometry const& geom, int nPhi, int nEta, double etaBinSize) const final {
    return std::unique_ptr<DetIdAssociator>(new HcalDetIdAssociator(hcalReg_, nPhi, nEta, etaBinSize, &geom));
  }

  const int hcalReg_;
};

#endif
