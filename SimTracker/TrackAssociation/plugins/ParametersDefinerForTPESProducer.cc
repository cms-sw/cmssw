#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"

// system include files
#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

ParametersDefinerForTPESProducer::ParametersDefinerForTPESProducer(const edm::ParameterSet &iConfig) : pset_(iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  std::string myName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myName);

  // now do what ever other initialization is needed
  // conf_=iConfig;
}

ParametersDefinerForTPESProducer::~ParametersDefinerForTPESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
ParametersDefinerForTPESProducer::ReturnType ParametersDefinerForTPESProducer::produce(
    const TrackAssociatorRecord &iRecord) {
  return std::make_unique<ParametersDefinerForTP>(pset_);
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(ParametersDefinerForTPESProducer);
