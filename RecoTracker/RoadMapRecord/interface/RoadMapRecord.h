#ifndef RoadMapRecord_RoadMapRecord_h
#define RoadMapRecord_RoadMapRecord_h

//
// Package:         RecoTracker/RoadMapRecord
// Class:           RoadMapRecord
// 
// Description:     record for roads
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Feb  4 19:15:56 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/02/05 19:22:45 $
// $Revision: 1.1 $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "RecoTracker/RingRecord/interface/RingRecord.h"

class RoadMapRecord : public edm::eventsetup::DependentRecordImplementation<RoadMapRecord,
									    boost::mpl::vector<RingRecord> > {};
#endif
