#ifndef TrackAssociator_EcalDetIdAssociator_h
#define TrackAssociator_EcalDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      EcalDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: EcalDetIdAssociator.h,v 1.1 2006/06/24 04:56:07 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"

class EcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   EcalDetIdAssociator():CaloDetIdAssociator(180,150,0.04){};
 protected:

   virtual std::set<DetId> getASetOfValidDetIds(){
      std::set<DetId> setOfValidIds;
      std::vector<DetId> vectOfValidIds = geometry_->getValidDetIds(DetId::Ecal, 1);//EB
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

      /*
         vectOfValidIds.clear();
         vectOfValidIds = geometry_->getValidDetIds(DetId::Ecal, 2);//EE
         for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);
       */

      return setOfValidIds;
   };

};
#endif
