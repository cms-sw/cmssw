#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>
#include <string>

using namespace edm;

class TransientTrackBuilderTest : public edm::EDAnalyzer {
public:
  TransientTrackBuilderTest(const edm::ParameterSet& pset) { conf_ = pset; }

  ~TransientTrackBuilderTest() override {}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    using namespace std;

    cout << " Asking for the TransientTrackBuilder with name TransientTrackBuilder\n";
    edm::ESHandle<TransientTrackBuilder> theB;
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);

    cout << " Got a " << typeid(*theB).name() << endl;
    cout << "Field at origin (in Testla): " << (*theB).field()->inTesla(GlobalPoint(0., 0., 0.)) << endl;
  }

private:
  edm::ParameterSet conf_;
};
