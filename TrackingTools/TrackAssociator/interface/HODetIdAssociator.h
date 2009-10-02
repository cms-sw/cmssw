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
// $Id: HODetIdAssociator.h,v 1.2 2007/10/08 13:04:31 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"

class HODetIdAssociator: public CaloDetIdAssociator{
 public:
   HODetIdAssociator():CaloDetIdAssociator(72,30,0.087){};

   HODetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

 protected:

   virtual std::set<DetId> getASetOfValidDetIds() const {
      std::set<DetId> setOfValidIds;
      const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Hcal, 3);//HO
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);
      return setOfValidIds;
   };

};
#endif
