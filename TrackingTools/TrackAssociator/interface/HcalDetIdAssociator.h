#ifndef TrackAssociator_HcalDetIdAssociator_h
#define TrackAssociator_HcalDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      HcalDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: HcalDetIdAssociator.h,v 1.2 2007/10/08 13:04:31 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"

class HcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   HcalDetIdAssociator():CaloDetIdAssociator(72, 70 ,0.087){};

   HcalDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

 protected:

   virtual std::set<DetId> getASetOfValidDetIds() const {
      std::set<DetId> setOfValidIds;
      const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Hcal, 1);//HB
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

//      vectOfValidIds.clear();
      const std::vector<DetId>& vectOfValidIdsHE = geometry_->getValidDetIds(DetId::Hcal, 2);//HE
      for(std::vector<DetId>::const_iterator it = vectOfValidIdsHE.begin(); it != vectOfValidIdsHE.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };

};
#endif
