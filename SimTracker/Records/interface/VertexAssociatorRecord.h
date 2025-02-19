#ifndef SimTracker_VertexAssociatorRecord_h
#define SimTracker_VertexAssociatorRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "boost/mpl/vector.hpp"

class VertexAssociatorRecord : public edm::eventsetup::DependentRecordImplementation<VertexAssociatorRecord, 
			      boost::mpl::vector<IdealMagneticFieldRecord> > {};

#endif
