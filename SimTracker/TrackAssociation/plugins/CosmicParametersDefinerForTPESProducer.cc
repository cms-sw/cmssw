#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"

// system include files
#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

CosmicParametersDefinerForTPESProducer::CosmicParametersDefinerForTPESProducer(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  std::string myName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myName);

  // now do what ever other initialization is needed
  // conf_=iConfig;
}

CosmicParametersDefinerForTPESProducer::~CosmicParametersDefinerForTPESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CosmicParametersDefinerForTPESProducer::ReturnType CosmicParametersDefinerForTPESProducer::produce(
    const TrackAssociatorRecord &iRecord) {
  return std::make_unique<CosmicParametersDefinerForTP>();
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CosmicParametersDefinerForTPESProducer);
