#ifndef RECOTRACKER_ROADMAPESSOURCE_H
#define RECOTRACKER_ROADMAPESSOURCE_H

//
// Package:         RecoTracker/RoadMapESSource
// Class:           RoadMapESSource
// 
// Description:     Reads in ASCII dump of Roads object
//                  and provides it to the event.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/14 22:00:00 $
// $Revision: 1.1 $
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

#endif
