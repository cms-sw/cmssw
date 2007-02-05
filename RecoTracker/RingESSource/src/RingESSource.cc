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
// $Date: 2006/01/15 00:59:13 $
// $Revision: 1.2 $
//

#include "RecoTracker/RingESSource/interface/RingESSource.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

RingESSource::RingESSource(const edm::ParameterSet& iConfig) : 
  fileName_((iConfig.getParameter<edm::FileInPath>("InputFileName")).fullPath()) {

  setWhatProduced(this);

  findingRecord<RingRecord>();
}


RingESSource::~RingESSource()
{
 
}

RingESSource::ReturnType
RingESSource::produce(const RingRecord& iRecord)
{
  using namespace edm::es;

  std::auto_ptr<Rings> rings(new Rings(fileName_));

  return rings ;
}

void RingESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(RingESSource);

