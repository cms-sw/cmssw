//
// Package:         RecoTracker/RingESSource
// Class:           RingESSource
// 
// Description:     Reads in ASCII dump of Rings
//                  and provides it to the event.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Oct  5 01:35:14 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/30 02:49:35 $
// $Revision: 1.4 $
//

#include "RecoTracker/RingESSource/interface/RingESSource.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

RingESSource::RingESSource(const edm::ParameterSet& iConfig) : fileName_((iConfig.getParameter<edm::FileInPath>("InputFileName")).fullPath()) {
  
  std::string componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
    
  findingRecord<RingRecord>();
  
  rings_ = 0;
}


RingESSource::~RingESSource()
{
   
}

RingESSource::ReturnType
RingESSource::produce(const RingRecord& iRecord)
{
  using namespace edm::es;

  rings_ = new Rings(fileName_);
  
  std::auto_ptr<Rings> rings(rings_);

  return rings ;
}

void RingESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(RingESSource);

