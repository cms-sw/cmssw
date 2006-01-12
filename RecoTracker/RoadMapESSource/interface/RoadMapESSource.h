// -*- C++ -*-
//
// Package:    RoadMapESSource
// Class:      RoadMapESSource
// 
/**\class RoadMapESSource RoadMapESSource.h RecoTracker/RoadMapESSource/interface/RoadMapESSource.h

Description: reads in RoadMap from ascii file

*/
//
// Original Author:  Oliver Gutsche
//         Created:  Wed Nov 16 14:22:12 CST 2005
// $Id: RoadMapESSource.h,v 1.1.1.1 2005/11/29 21:14:41 gutsche Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//
// class declaration
//

class RoadMapESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  RoadMapESSource(const edm::ParameterSet&);
  ~RoadMapESSource();

  typedef std::auto_ptr<Roads> ReturnType;

  ReturnType produce(const TrackerDigiGeometryRecord&);

 protected:

  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &); 

 private:

  // ----------member data ---------------------------
  std::string fileName_;
  unsigned int verbosity_;
};
