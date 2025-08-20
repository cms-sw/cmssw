/** \class SmartPropagatorESProducer
 *  ES producer needed to put the SmartPropagator inside the EventSetup
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class SmartPropagatorESProducer : public edm::ESProducer {
public:
  /// Constructor
  SmartPropagatorESProducer(const edm::ParameterSet&);

  /// Destructor
  ~SmartPropagatorESProducer() override = default;

  // Operations
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord&);

  // fillDescriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  PropagationDirection thePropagationDirection;
  double theEpsilon;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> trackerToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> muonToken_;
};

using namespace edm;
using namespace std;

SmartPropagatorESProducer::SmartPropagatorESProducer(const ParameterSet& parameterSet) {
  string myname = parameterSet.getParameter<string>("ComponentName");

  string propDir = parameterSet.getParameter<string>("PropagationDirection");

  if (propDir == "oppositeToMomentum")
    thePropagationDirection = oppositeToMomentum;
  else if (propDir == "alongMomentum")
    thePropagationDirection = alongMomentum;
  else if (propDir == "anyDirection")
    thePropagationDirection = anyDirection;
  else
    throw cms::Exception("SmartPropagatorESProducer") << "Wrong fit direction chosen in SmartPropagatorESProducer";

  theEpsilon = parameterSet.getParameter<double>("Epsilon");

  auto cc = setWhatProduced(this, myname);
  magToken_ = cc.consumes();
  trackerToken_ = cc.consumes(edm::ESInputTag("", parameterSet.getParameter<string>("TrackerPropagator")));
  muonToken_ = cc.consumes(edm::ESInputTag("", parameterSet.getParameter<string>("MuonPropagator")));
}

std::unique_ptr<Propagator> SmartPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  return std::make_unique<SmartPropagator>(
      iRecord.get(trackerToken_), iRecord.get(muonToken_), &iRecord.get(magToken_), thePropagationDirection, theEpsilon);
}

void SmartPropagatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("ComponentName", "SmartPropagator");
  desc.add<string>("PropagationDirection", "alongMomentum");
  desc.add<double>("Epsilon", 5.0);
  desc.add<string>("TrackerPropagator", "PropagatorWithMaterial");
  desc.add<string>("MuonPropagator", "SteppingHelixPropagatorAlong");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(SmartPropagatorESProducer);
