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
// $Id: EcalDetIdAssociator.h,v 1.6 2009/09/06 16:36:52 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
class EcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   EcalDetIdAssociator():CaloDetIdAssociator(360,300,0.02){};

   EcalDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

 protected:

   virtual std::set<DetId> getASetOfValidDetIds() const {
      std::set<DetId> setOfValidIds;
      const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Ecal, EcalBarrel);//EB
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

//      vectOfValidIds.clear();
      const std::vector<DetId>& vectOfValidIdsEE = geometry_->getValidDetIds(DetId::Ecal, EcalEndcap);//EE
      for(std::vector<DetId>::const_iterator it = vectOfValidIdsEE.begin(); it != vectOfValidIdsEE.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };

};
#endif
