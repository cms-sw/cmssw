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
  BeamHaloPropagatorESProducer(const edm::ParameterSet &);

  /// Destructor
  ~BeamHaloPropagatorESProducer() override;

  // Operations
  std::unique_ptr<Propagator> produce(const TrackingComponentsRecord &);

private:
  PropagationDirection thePropagationDirection;
  std::string myname;
  std::string theEndCapTrackerPropagatorName;
  std::string theCrossingTrackerPropagatorName;
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

  setWhatProduced(this, myname);
}

BeamHaloPropagatorESProducer::~BeamHaloPropagatorESProducer() {}

std::unique_ptr<Propagator> BeamHaloPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord) {
  ESHandle<MagneticField> magField;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magField);

  ESHandle<Propagator> endcapPropagator;
  iRecord.get(theEndCapTrackerPropagatorName, endcapPropagator);

  ESHandle<Propagator> crossPropagator;
  iRecord.get(theCrossingTrackerPropagatorName, crossPropagator);

  LogDebug("BeamHaloPropagator") << "Creating a BeamHaloPropagator: " << myname
                                 << "\n with EndCap Propagator: " << theEndCapTrackerPropagatorName
                                 << "\n with Crossing Propagator: " << theCrossingTrackerPropagatorName;

  return std::make_unique<BeamHaloPropagator>(*endcapPropagator, *crossPropagator, &*magField, thePropagationDirection);
}

DEFINE_FWK_EVENTSETUP_MODULE(BeamHaloPropagatorESProducer);
