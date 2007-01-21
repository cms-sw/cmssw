#ifndef TrackAssociator_HODetIdAssociator_h
#define TrackAssociator_HODetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      HODetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: HODetIdAssociator.h,v 1.3 2006/09/01 17:21:41 jribnik Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"

class HODetIdAssociator: public CaloDetIdAssociator{
 public:
   HODetIdAssociator():CaloDetIdAssociator(72,30,0.087){};
 protected:

   virtual std::set<DetId> getASetOfValidDetIds(){
      std::set<DetId> setOfValidIds;
      std::vector<DetId> vectOfValidIds = geometry_->getValidDetIds(DetId::Hcal, 3);//HO
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);
      return setOfValidIds;
   };

};
#endif
