#ifndef SimTracker_TrackAssociatorRecord_h
#define SimTracker_TrackAssociatorRecord_h

/** \class TrackAssociatorRecord
 *  Record of Tracs Associators
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TrackAssociatorRecord
    : public edm::eventsetup::DependentRecordImplementation<
          TrackAssociatorRecord,
          edm::mpl::Vector<IdealMagneticFieldRecord, TrackingComponentsRecord, GlobalTrackingGeometryRecord>> {};

#endif
