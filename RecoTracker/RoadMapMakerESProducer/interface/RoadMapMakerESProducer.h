#ifndef RECOTRACKER_ROADMAPMAKERESPRODUCER_H
#define RECOTRACKER_ROADMAPMAKERESPRODUCER_H

//
// Package:         RecoTracker/RoadMapMakerESProducer
// Class:           RoadMapMakerESProducer
// 
// Description:     Uses the RoadMaker object to construct
//                  and provide a Roads object.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Jan 12 21:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/15 01:00:30 $
// $Revision: 1.2 $
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

//
// class decleration
//

class RoadMapMakerESProducer : public edm::ESProducer {

 public:

  RoadMapMakerESProducer(const edm::ParameterSet&);
  ~RoadMapMakerESProducer();

  typedef std::auto_ptr<Roads> ReturnType;

  ReturnType produce(const TrackerDigiGeometryRecord&);

 private:
  // ----------member data ---------------------------
  unsigned int verbosity_;
  bool writeOut_;
  std::string fileName_;
  bool writeOutOldStyle_;
  std::string fileNameOldStyle_;
  bool writeOutTrackerAsciiDump_;
  std::string fileNameTrackerAsciiDump_;
};

#endif
