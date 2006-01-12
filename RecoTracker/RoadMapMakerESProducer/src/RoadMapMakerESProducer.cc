// -*- C++ -*-
//
// Package:    RoadMapMakerESProducer
// Class:      RoadMapMakerESProducer
// 
//

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMapMakerESProducer.h"

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMaker.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

//
// constructors and destructor
//
RoadMapMakerESProducer::RoadMapMakerESProducer(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  //now do what ever other initialization is needed
  verbosity_         = iConfig.getUntrackedParameter<int>("VerbosityLevel",0);
  writeOut_          = iConfig.getUntrackedParameter<bool>("WriteOutRoadMapToAsciiFile",false);
  fileName_          = iConfig.getUntrackedParameter<std::string>("RoadMapAsciiFile","");
  writeOutOldStyle_  = iConfig.getUntrackedParameter<bool>("WriteOutRoadMapToAsciiFileOldStyle",false);
  fileNameOldStyle_  = iConfig.getUntrackedParameter<std::string>("RoadMapAsciiFileOldStyle","");
}


RoadMapMakerESProducer::~RoadMapMakerESProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RoadMapMakerESProducer::ReturnType
RoadMapMakerESProducer::produce(const TrackerDigiGeometryRecord& iRecord)
{

  // get geometry
  edm::ESHandle<TrackingGeometry> trackingGeometryHandle;
  iRecord.get(trackingGeometryHandle);
  const TrackingGeometry& tracker(*trackingGeometryHandle);

  RoadMaker maker(tracker,verbosity_);

  Roads *roads = maker.getRoads();
  
  ReturnType pRoads(roads) ;

  if ( writeOut_ ) {
    roads->dump(fileName_);
  }

  if ( writeOutOldStyle_ ) {
    maker.dumpOldStyle(fileNameOldStyle_);
  }

  return pRoads ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RoadMapMakerESProducer)
