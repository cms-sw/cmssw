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
// $Date: 2007/03/15 20:17:22 $
// $Revision: 1.6 $
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMaker.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

class RoadMapMakerESProducer : public edm::ESProducer {

 public:

  RoadMapMakerESProducer(const edm::ParameterSet&);
  ~RoadMapMakerESProducer();

  typedef std::auto_ptr<Roads> ReturnType;

  ReturnType produce(const RoadMapRecord&);

 private:

  bool writeOut_;
  std::string fileName_;

  RoadMaker::GeometryStructure geometryStructure_;
  RoadMaker::SeedingType       seedingType_;

  std::string ringsLabel_;

  Roads *roads_;
  
};

#endif
