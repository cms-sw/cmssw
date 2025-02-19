#ifndef RingRecord_RingRecord_h
#define RingRecord_RingRecord_h

//
// Package:         RecoTracker/RingRecord
// Class:           RingRecord
// 
// Description:     record for rings
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Feb  4 19:15:56 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/02/05 19:10:03 $
// $Revision: 1.1 $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 

class RingRecord : public edm::eventsetup::DependentRecordImplementation<RingRecord,
									 boost::mpl::vector<TrackerDigiGeometryRecord> > {};

#endif
