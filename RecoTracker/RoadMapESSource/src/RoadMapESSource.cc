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
// $Date: 2007/03/30 02:49:38 $
// $Revision: 1.7 $
//

#include "RecoTracker/RoadMapESSource/interface/RoadMapESSource.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"
#include "RecoTracker/RingRecord/interface/Rings.h"

//
// constructors and destructor
//
RoadMapESSource::RoadMapESSource(const edm::ParameterSet& iConfig) : 
  fileName_((iConfig.getParameter<edm::FileInPath>("InputFileName")).fullPath())
{

  std::string componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);

  ringsLabel_ = iConfig.getParameter<std::string>("RingsLabel");

  findingRecord<RoadMapRecord>();
  
  roads_ = 0;
}


RoadMapESSource::~RoadMapESSource()
{
}

RoadMapESSource::ReturnType
RoadMapESSource::produce(const RoadMapRecord& iRecord)
{

  // get rings
  edm::ESHandle<Rings> ringHandle;
  iRecord.getRecord<RingRecord>().get(ringsLabel_, ringHandle);
  const Rings *rings = ringHandle.product();

  roads_ = new Roads(fileName_,rings);
  
  std::auto_ptr<Roads> roads(roads_);

  return roads ;
}

void RoadMapESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(RoadMapESSource);

