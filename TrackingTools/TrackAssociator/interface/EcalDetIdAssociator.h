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
// $Id: EcalDetIdAssociator.h,v 1.8 2010/02/18 01:21:50 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
class EcalDetIdAssociator: public CaloDetIdAssociator{
 public:
   EcalDetIdAssociator():CaloDetIdAssociator(360,300,0.02){};

   EcalDetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

   virtual const char* name() const { return "ECAL"; }

 protected:

   virtual const unsigned int getNumberOfSubdetectors() const { return 2;}
   virtual const std::vector<DetId>& getValidDetIds(unsigned int subDetectorIndex) const {
     if ( subDetectorIndex == 0 )
       return geometry_->getValidDetIds(DetId::Ecal, EcalBarrel);//EB
     else
       return geometry_->getValidDetIds(DetId::Ecal, EcalEndcap);//EE
   };

};
#endif
