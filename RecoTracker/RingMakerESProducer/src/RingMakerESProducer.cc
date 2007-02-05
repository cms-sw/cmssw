//
// Package:         RecoTracker/RingMakerESProducer
// Class:           RingMakerESProducer
// 
// Description:     Uses the RingMaker object to construct
//                  and provide a Rings object.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Oct  3 23:51:34 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/03 22:23:12 $
// $Revision: 1.3 $
//

#include "RecoTracker/RingMakerESProducer/interface/RingMakerESProducer.h"

#include "RecoTracker/RingMakerESProducer/interface/RingMaker.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

RingMakerESProducer::RingMakerESProducer(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);

  writeOut_                  = iConfig.getUntrackedParameter<bool>("WriteOutRingsToAsciiFile",false);
  fileName_                  = iConfig.getUntrackedParameter<std::string>("RingAsciiFileName","");
  dumpDetIds_                = iConfig.getUntrackedParameter<bool>("DumpDetIds",false);
  detIdsDumpFileName_        = iConfig.getUntrackedParameter<std::string>("DetIdsDumpFileName","");
  configuration_             = iConfig.getUntrackedParameter<std::string>("Configuration","FULL");

}


RingMakerESProducer::~RingMakerESProducer()
{
}


RingMakerESProducer::ReturnType
RingMakerESProducer::produce(const RingRecord& iRecord)
{

  // get geometry
  edm::ESHandle<TrackerGeometry> trackingGeometryHandle;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(trackingGeometryHandle);
  const TrackerGeometry *tracker = trackingGeometryHandle.product();

  RingMaker maker(tracker,configuration_);

  if ( dumpDetIds_ ) {
    maker.dumpDetIdsIntoFile(detIdsDumpFileName_);
  }

  Rings *rings = maker.getRings();
  
  ReturnType pRings(rings) ;

  if ( writeOut_ ) {
    rings->dump(fileName_);
  }

  return pRings ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(RingMakerESProducer);
