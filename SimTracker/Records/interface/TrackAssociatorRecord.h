#ifndef SimTracker_TrackAssociatorRecord_h
#define SimTracker_TrackAssociatorRecord_h

/** \class TrackAssociatorRecord
 *  Record of Tracs Associators
 *
 *  $Date: 2007/07/13 12:17:42 $
 *  $Revision: 1.3 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "boost/mpl/vector.hpp"

class TrackAssociatorRecord : public edm::eventsetup::DependentRecordImplementation<TrackAssociatorRecord,
  boost::mpl::vector<IdealMagneticFieldRecord,
  TrackingComponentsRecord,
GlobalTrackingGeometryRecord
> > {};

#endif
