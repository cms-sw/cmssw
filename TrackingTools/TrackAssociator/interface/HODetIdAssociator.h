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
// $Id: HODetIdAssociator.h,v 1.5 2010/02/18 15:45:41 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
class HODetIdAssociator: public CaloDetIdAssociator{
 public:
   HODetIdAssociator():CaloDetIdAssociator(72,30,0.087){};

   HODetIdAssociator(const edm::ParameterSet& pSet):CaloDetIdAssociator(pSet){};

   virtual const char* name() const { return "HO"; }

 protected:

   const std::vector<DetId>& getValidDetIds(unsigned int subDectorIndex) const
     {
       if ( subDectorIndex!=0 ) cms::Exception("FatalError") << 
	 "HO sub-dectors are all handle as one sub-system, but subDetectorIndex is not zero.\n";
       return geometry_->getValidDetIds(DetId::Hcal, HcalOuter);
     }
};
#endif
