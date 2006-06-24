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
// $Id: EcalDetIdAssociator.h,v 1.1 2006/06/09 17:30:20 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"

class EcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   EcalDetIdAssociator():CaloDetIdAssociator(180,150,0.04){};
 protected:
   virtual std::set<DetId> getASetOfValidDetIds(){
      std::set<DetId> validIds;
      // EB 3-1
      uint32_t detIdPrefix = ((3<<3)+1) << 25; 
      for(uint32_t z=0; z<=1; z++) // eta sign
	for(uint32_t i=1; i<=360;i++) // phi index range
	  for(uint32_t j=1; j<=85;j++){ //eta index range
	     uint32_t rawDetId = detIdPrefix+i+(j<<9)+(z<<16);
	     validIds.insert(DetId(rawDetId));
	  }
/*      
      // EE 3-2
      detIdPrefix = ((3<<3)+2) << 25; 
      for(uint32_t z=0; z<=1; z++) // eta sign
	for(uint32_t i=1; i<=100;i++) // X index range
	  for(uint32_t j=1; j<=100;j++){ //Y index range
	     uint32_t rawDetId = detIdPrefix+i+(j<<7)+(z<<14);
	     validIds.insert(DetId(rawDetId));
	  }
*/
      return validIds;

   }
};
#endif
