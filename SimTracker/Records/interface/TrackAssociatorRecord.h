#ifndef SimTracker_TrackAssociatorRecord_h
#define SimTracker_TrackAssociatorRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "boost/mpl/vector.hpp"

class TrackAssociatorRecord : public edm::eventsetup::DependentRecordImplementation<TrackAssociatorRecord, 
			      boost::mpl::vector<IdealMagneticFieldRecord> > {};

#endif
