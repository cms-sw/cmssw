#ifndef RecoTracker_Record_CkfComponentsRecord_h
#define RecoTracker_Record_CkfComponentsRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h" // FIXME should be in the dependencies of the SiPixelQualityRcd


#include "boost/mpl/vector.hpp"


class CkfComponentsRecord : public edm::eventsetup::DependentRecordImplementation<CkfComponentsRecord,
			    boost::mpl::vector<TrackerDigiGeometryRecord,
                                               TkPixelCPERecord,
                                               TkStripCPERecord,
			                       TransientRecHitRecord,
                                               TrackingComponentsRecord,
			                       TrackerRecoGeometryRecord,
                                               TrackerTopologyRcd,
                                               SiStripQualityRcd,
					       SiStripDetCablingRcd,
                                               SiStripNoisesRcd,
                                               SiStripRegionCablingRcd,
                                               SiPixelQualityRcd,
                                               SiPixelFedCablingMapRcd> > {};

#endif 

