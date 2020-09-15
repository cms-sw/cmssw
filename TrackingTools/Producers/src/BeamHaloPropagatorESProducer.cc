/** \class BeamHaloPropagatorESProducer
 *  ES producer needed to put the BeamHaloPropagator inside the EventSetup
 *
 *  \author Jean-Roch VLIMANT UCSB
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/GeomPropagators/interface/BeamHaloPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

class BeamHaloPropagatorESProducer : public edm::ESProducer {
public:
  /// Constructor
  BeamHaloPropagatorESProducer(const edm::ParameterSet&);

  /// Destructor
  ~BeamHaloPropagatorESProducer() override;

  // Operations
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord&);

private:
  PropagationDirection thePropagationDirection;
  std::string myname;
  std::string theEndCapTrackerPropagatorName;
  std::string theCrossingTrackerPropagatorName;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> endcapToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> crossToken_;
};

using namespace edm;
using namespace std;

BeamHaloPropagatorESProducer::BeamHaloPropagatorESProducer(const ParameterSet& parameterSet) {
  myname = parameterSet.getParameter<string>("ComponentName");

  string propDir = parameterSet.getParameter<string>("PropagationDirection");

  if (propDir == "oppositeToMomentum")
    thePropagationDirection = oppositeToMomentum;
  else if (propDir == "alongMomentum")
    thePropagationDirection = alongMomentum;
  else if (propDir == "anyDirection")
    thePropagationDirection = anyDirection;
  else
    throw cms::Exception("BeamHaloPropagatorESProducer")
        << "Wrong fit direction (" << propDir << ")chosen in BeamHaloPropagatorESProducer";

  theEndCapTrackerPropagatorName = parameterSet.getParameter<string>("EndCapTrackerPropagator");
  theCrossingTrackerPropagatorName = parameterSet.getParameter<string>("CrossingTrackerPropagator");

  auto cc = setWhatProduced(this, myname);
  magToken_ = cc.consumes();
  endcapToken_ = cc.consumes(edm::ESInputTag(""s, theEndCapTrackerPropagatorName));
  crossToken_ = cc.consumes(edm::ESInputTag(""s, theCrossingTrackerPropagatorName));
}

BeamHaloPropagatorESProducer::~BeamHaloPropagatorESProducer() {}

std::unique_ptr<Propagator> BeamHaloPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  LogDebug("BeamHaloPropagator") << "Creating a BeamHaloPropagator: " << myname
                                 << "\n with EndCap Propagator: " << theEndCapTrackerPropagatorName
                                 << "\n with Crossing Propagator: " << theCrossingTrackerPropagatorName;

  return std::make_unique<BeamHaloPropagator>(
      iRecord.get(endcapToken_), iRecord.get(crossToken_), &iRecord.get(magToken_), thePropagationDirection);
}

DEFINE_FWK_EVENTSETUP_MODULE(BeamHaloPropagatorESProducer);
