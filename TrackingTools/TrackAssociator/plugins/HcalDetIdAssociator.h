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
// $Id: HcalDetIdAssociator.h,v 1.4 2010/02/18 14:35:48 dmytro Exp $
//
//

#include "CaloDetIdAssociator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
class HcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   HcalDetIdAssociator():CaloDetIdAssociator(72, 70 ,0.087){};

   HcalDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};
   
   virtual const char* name() const { return "HCAL"; }

 protected:

   virtual const unsigned int getNumberOfSubdetectors() const { return 2;}
   virtual const std::vector<DetId>& getValidDetIds(unsigned int subDetectorIndex) const {
     if ( subDetectorIndex == 0 )
       return geometry_->getValidDetIds(DetId::Hcal, HcalBarrel);//HB
     else
       return geometry_->getValidDetIds(DetId::Hcal, HcalEndcap);//HE
   };
};
#endif
