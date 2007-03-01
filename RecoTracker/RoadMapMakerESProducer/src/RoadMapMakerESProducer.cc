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
// $Author: noeding $
// $Date: 2007/02/23 00:48:52 $
// $Revision: 1.7 $
//

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMapMakerESProducer.h"

#include "RecoTracker/RingRecord/interface/Rings.h"
#include "RecoTracker/RingRecord/interface/RingRecord.h"

RoadMapMakerESProducer::RoadMapMakerESProducer(const edm::ParameterSet& iConfig)
{

  std::string componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);

  ringsLabel_                = iConfig.getParameter<std::string>("RingsLabel");

  writeOut_                  = iConfig.getUntrackedParameter<bool>("WriteOutRoadMapToAsciiFile",false);
  fileName_                  = iConfig.getUntrackedParameter<std::string>("RoadMapAsciiFile","");

  std::string tmp_string         = iConfig.getParameter<std::string>("GeometryStructure");

  if ( tmp_string == "MTCC" ) {
    geometryStructure_ = RoadMaker::MTCC;
  } else if ( tmp_string == "TIF" ) {
    geometryStructure_ = RoadMaker::TIF;
  } else if ( tmp_string == "TIFTOB" ) {
    geometryStructure_ = RoadMaker::TIFTOB;
  } else if ( tmp_string == "TIFTIB" ) {
    geometryStructure_ = RoadMaker::TIFTIB;
  }else if ( tmp_string == "TIFTIBTOB" ) {
    geometryStructure_ = RoadMaker::TIFTIBTOB;
  }else if ( tmp_string == "TIFTOBTEC" ) {
    geometryStructure_ = RoadMaker::TIFTOBTEC;
  } else if ( tmp_string == "FullDetector" ) {
    geometryStructure_ = RoadMaker::FullDetector;
  } else {
    geometryStructure_ = RoadMaker::FullDetector;
  }

  tmp_string         = iConfig.getParameter<std::string>("SeedingType");

  if ( tmp_string == "TwoRingSeeds" ) {
    seedingType_ = RoadMaker::TwoRingSeeds;
  } else if ( tmp_string == "FourRingSeeds" ) {
    seedingType_ = RoadMaker::FourRingSeeds;
  } else {
    seedingType_ = RoadMaker::FourRingSeeds;
  }

}


RoadMapMakerESProducer::~RoadMapMakerESProducer()
{
 
}


RoadMapMakerESProducer::ReturnType
RoadMapMakerESProducer::produce(const RoadMapRecord& iRecord)
{

  // get rings
  edm::ESHandle<Rings> ringHandle;
  iRecord.getRecord<RingRecord>().get(ringsLabel_, ringHandle);
  const Rings *rings = ringHandle.product();

  RoadMaker maker(rings,
		  geometryStructure_,
		  seedingType_);

  Roads *roads = maker.getRoads();
  
  ReturnType pRoads(roads) ;

  if ( writeOut_ ) {
    roads->dump(fileName_);
  }

  return pRoads ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RoadMapMakerESProducer);
