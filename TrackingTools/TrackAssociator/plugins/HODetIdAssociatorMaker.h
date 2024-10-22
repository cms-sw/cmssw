#ifndef TrackingTools_TrackAssociator_HODetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_HODetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     HODetIdAssociatorMaker
//
/**\class HODetIdAssociatorMaker HODetIdAssociatorMaker.h "HODetIdAssociatorMaker.h"

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
#include "HODetIdAssociator.h"

class HODetIdAssociatorMaker : public CaloDetIdAssociatorMaker {
public:
  using CaloDetIdAssociatorMaker::CaloDetIdAssociatorMaker;

private:
  std::unique_ptr<DetIdAssociator> make(CaloGeometry const& geom, int nPhi, int nEta, double etaBinSize) const final {
    return std::unique_ptr<DetIdAssociator>(new HODetIdAssociator(nPhi, nEta, etaBinSize, &geom));
  }
};

#endif
