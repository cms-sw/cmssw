//
// Original Author:  Matt Rudolph
//         Created:  Sat Mar 28 20:13:08 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//
// class decleration
//

class TSCBLBuilderWithPropagatorESProducer : public edm::ESProducer {
public:
  TSCBLBuilderWithPropagatorESProducer(const edm::ParameterSet&);
  ~TSCBLBuilderWithPropagatorESProducer() override;

  typedef std::unique_ptr<TrajectoryStateClosestToBeamLineBuilder> ReturnType;

  ReturnType produce(const TrackingComponentsRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken_;
  const std::string myName_;
  const std::string propName_;
};

//
// constructors and destructor
//
TSCBLBuilderWithPropagatorESProducer::TSCBLBuilderWithPropagatorESProducer(const edm::ParameterSet& p)
    : myName_(p.getParameter<std::string>("ComponentName")), propName_(p.getParameter<std::string>("Propagator")) {
  auto cc = setWhatProduced(this, myName_);
  //now do what ever other initialization is needed
  propToken_ = cc.consumes(edm::ESInputTag{"", propName_});
}

TSCBLBuilderWithPropagatorESProducer::~TSCBLBuilderWithPropagatorESProducer() = default;

//
// member functions
//
// ------------ method called to produce the data  ------------
TSCBLBuilderWithPropagatorESProducer::ReturnType TSCBLBuilderWithPropagatorESProducer::produce(
    const TrackingComponentsRecord& iRecord) {
  using namespace edm::es;

  const Propagator* pro = &iRecord.get(propToken_);
  auto pTSCBLBuilderWithPropagator = std::make_unique<TSCBLBuilderWithPropagator>(*pro);
  return pTSCBLBuilderWithPropagator;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TSCBLBuilderWithPropagatorESProducer);
