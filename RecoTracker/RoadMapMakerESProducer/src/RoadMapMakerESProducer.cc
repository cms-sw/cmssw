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
// $Author: wmtan $
// $Date: 2006/10/27 01:35:39 $
// $Revision: 1.5 $
//

#include "RecoTracker/RoadMapMakerESProducer/interface/RoadMapMakerESProducer.h"

#include "RecoTracker/RingRecord/interface/Rings.h"
#include "RecoTracker/RingRecord/interface/RingRecord.h"

RoadMapMakerESProducer::RoadMapMakerESProducer(const edm::ParameterSet& iConfig)
{

  setWhatProduced(this);

  writeOut_                  = iConfig.getUntrackedParameter<bool>("WriteOutRoadMapToAsciiFile",false);
  fileName_                  = iConfig.getUntrackedParameter<std::string>("RoadMapAsciiFile","");

  std::string tmp_string         = iConfig.getUntrackedParameter<std::string>("GeometryStructure","FullDetector");

  if ( tmp_string == "MTCC" ) {
    geometryStructure_ = RoadMaker::MTCC;
  } else if ( tmp_string == "TIFTOB" ) {
    geometryStructure_ = RoadMaker::TIFTOB;
  } else if ( tmp_string == "FullDetector" ) {
    geometryStructure_ = RoadMaker::FullDetector;
  } else {
    geometryStructure_ = RoadMaker::FullDetector;
  }

  tmp_string         = iConfig.getUntrackedParameter<std::string>("SeedingType","FourRingSeeds");

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
  iRecord.getRecord<RingRecord>().get(ringHandle);
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
