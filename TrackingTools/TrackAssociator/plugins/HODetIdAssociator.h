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
//
//

#include "CaloDetIdAssociator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
class HODetIdAssociator: public CaloDetIdAssociator{
 public:
   HODetIdAssociator():CaloDetIdAssociator(72,30,0.087){};

   HODetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

   const char* name() const override { return "HO"; }

 protected:

   void getValidDetIds(unsigned int subDectorIndex, std::vector<DetId>& validIds) const override
     {
       if ( subDectorIndex!=0 ) cms::Exception("FatalError") << 
	 "HO sub-dectors are all handle as one sub-system, but subDetectorIndex is not zero.\n";
       validIds = geometry_->getValidDetIds(DetId::Hcal, HcalOuter);
     }
};
#endif
