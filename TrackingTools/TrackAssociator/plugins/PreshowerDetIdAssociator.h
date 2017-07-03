#ifndef TrackAssociator_PreshowerDetIdAssociator_h
#define TrackAssociator_PreshowerDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      PreshowerDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//
//

#include "CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class PreshowerDetIdAssociator: public CaloDetIdAssociator{
 public:
   PreshowerDetIdAssociator():CaloDetIdAssociator(30,60,0.1){};

   PreshowerDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};
     
   const char* name() const override { return "Preshower"; }
 protected:

   void getValidDetIds(unsigned int subDetectorIndex, std::vector<DetId>& validIds) const override {
     if ( subDetectorIndex != 0 ) throw cms::Exception("FatalError") << "Preshower has only one sub-detector for geometry. Abort.";
     validIds = geometry_->getValidDetIds(DetId::Ecal, EcalPreshower);
   };

};
#endif
