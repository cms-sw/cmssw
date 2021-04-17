#ifndef RecoTracker_Record_TkMSParameterizationRecord_h
#define RecoTracker_Record_TkMSParameterizationRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TkMSParameterizationRecord : public edm::eventsetup::DependentRecordImplementation<
                                       TkMSParameterizationRecord,
                                       edm::mpl::Vector<TrackingComponentsRecord, NavigationSchoolRecord> > {};

#endif
