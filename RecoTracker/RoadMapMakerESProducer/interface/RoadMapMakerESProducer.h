// -*- C++ -*-
//
// Package:    RoadMapMakerESProducer
// Class:      RoadMapMakerESProducer
// 
/**\class RoadMapMakerESProducer RoadMapMakerESProducer.h RecoTracker/RoadMapMakerESProducer/interface/RoadMapMakerESProducer.h

Description: constructs RoadMap using the tracker geometry

*/
//
// Original Author:  Oliver Gutsche
//         Created:  Wed Nov 16 15:04:57 CST 2005
// $Id: RoadMapMakerESProducer.h,v 1.2 2005/12/08 23:18:54 gutsche Exp $
//
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
};

