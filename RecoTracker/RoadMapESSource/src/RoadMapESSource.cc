// -*- C++ -*-
//
// Package:    RoadMapESSource
// Class:      RoadMapESSource
// 
//

#include "RecoTracker/RoadMapESSource/interface/RoadMapESSource.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

//
// constructors and destructor
//
RoadMapESSource::RoadMapESSource(const edm::ParameterSet& iConfig) : 
  fileName_((iConfig.getParameter<edm::FileInPath>("InputFileName")).fullPath()),
  verbosity_(iConfig.getUntrackedParameter<int>("VerbosityLevel",0)) {

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  //now do what ever other initialization is needed
  findingRecord<TrackerDigiGeometryRecord>();
}


RoadMapESSource::~RoadMapESSource()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
RoadMapESSource::ReturnType
RoadMapESSource::produce(const TrackerDigiGeometryRecord& iRecord)
{
  using namespace edm::es;

  std::auto_ptr<Roads> roads(new Roads(fileName_,verbosity_));

  return roads ;
}

void RoadMapESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(RoadMapESSource)

