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
// $Id: PreshowerDetIdAssociator.h,v 1.4 2010/02/18 14:35:48 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class PreshowerDetIdAssociator: public CaloDetIdAssociator{
 public:
   PreshowerDetIdAssociator():CaloDetIdAssociator(30,60,0.1){};

   PreshowerDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};
     
   virtual const char* name() const { return "Preshower"; }
 protected:

   virtual const std::vector<DetId>& getValidDetIds(unsigned int subDetectorIndex) const {
     if ( subDetectorIndex != 0 ) throw cms::Exception("FatalError") << "Preshower has only one sub-detector for geometry. Abort.";
     return geometry_->getValidDetIds(DetId::Ecal, EcalPreshower);
   };

};
#endif
