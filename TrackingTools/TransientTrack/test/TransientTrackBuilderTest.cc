#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <memory>
#include <iostream>
#include <string>

using namespace edm;

class TransientTrackBuilderTest : public edm::one::EDAnalyzer<> {
public:
  TransientTrackBuilderTest(const edm::ParameterSet& pset)
      : ttkToken_(esConsumes(edm::ESInputTag{"", "TransientTrackBuilder"})) {}

  ~TransientTrackBuilderTest() = default;

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {
    using namespace std;

    edm::LogPrint("TrackerTrackBuilderTest")
        << " Asking for the TransientTrackBuilder with name TransientTrackBuilder\n";
    const TransientTrackBuilder* theB = &setup.getData(ttkToken_);

    edm::LogPrint("TrackerTrackBuilderTest") << " Got a " << typeid(*theB).name() << endl;
    edm::LogPrint("TrackerTrackBuilderTest")
        << "Field at origin (in Testla): " << (*theB).field()->inTesla(GlobalPoint(0., 0., 0.)) << endl;
  }

private:
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttkToken_;
};
