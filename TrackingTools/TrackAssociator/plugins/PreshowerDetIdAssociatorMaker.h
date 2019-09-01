#ifndef TrackingTools_TrackAssociator_PreshowerDetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_PreshowerDetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     PreshowerDetIdAssociatorMaker
//
/**\class PreshowerDetIdAssociatorMaker PreshowerDetIdAssociatorMaker.h "PreshowerDetIdAssociatorMaker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 16:18:21 GMT
//

// system include files

// user include files
#include "CaloDetIdAssociatorMaker.h"
#include "PreshowerDetIdAssociator.h"

// forward declarations

class PreshowerDetIdAssociatorMaker : public CaloDetIdAssociatorMaker {
public:
  using CaloDetIdAssociatorMaker::CaloDetIdAssociatorMaker;

private:
  std::unique_ptr<DetIdAssociator> make(CaloGeometry const& geom, int nPhi, int nEta, double etaBinSize) const final {
    return std::unique_ptr<DetIdAssociator>(new PreshowerDetIdAssociator(nPhi, nEta, etaBinSize, &geom));
  }
};

#endif
