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
// $Id: PreshowerDetIdAssociator.h,v 1.1.2.1 2009/07/01 10:04:16 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class PreshowerDetIdAssociator: public CaloDetIdAssociator{
 public:
   PreshowerDetIdAssociator():CaloDetIdAssociator(30,60,0.1){};

   PreshowerDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

 protected:

   virtual std::set<DetId> getASetOfValidDetIds() const {
      std::set<DetId> setOfValidIds;
      std::vector<DetId> vectOfValidIds = geometry_->getValidDetIds(DetId::Ecal, EcalPreshower);
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };

};
#endif
