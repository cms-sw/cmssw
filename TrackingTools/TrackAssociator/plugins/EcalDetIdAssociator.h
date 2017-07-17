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
//
//

#include "CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
class EcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   EcalDetIdAssociator():CaloDetIdAssociator(360,300,0.02){};

   EcalDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

   virtual const char* name() const override { return "ECAL"; }

 protected:

   virtual const unsigned int getNumberOfSubdetectors() const override { return 2;}
   virtual void getValidDetIds(unsigned int subDetectorIndex, std::vector<DetId>& validIds) const  override{
     if ( subDetectorIndex == 0 )
       validIds = geometry_->getValidDetIds(DetId::Ecal, EcalBarrel);//EB
     else
       validIds = geometry_->getValidDetIds(DetId::Ecal, EcalEndcap);//EE
   };

};
#endif
