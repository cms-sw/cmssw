#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <iostream>
#include <string>

using namespace edm;

class TTRHBuilderTest : public edm::EDAnalyzer {
public:
  TTRHBuilderTest(const edm::ParameterSet& pset) { conf_ = pset; }

  ~TTRHBuilderTest() override {}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    using namespace std;

    //     std::string cpeName = conf_.getParameter<std::string>("PixelCPE");
    //     cout <<" Asking for the CPE with name "<<cpeName<<endl;

    //     edm::ESHandle<PixelClusterParameterEstimator> theEstimator;
    //     setup.get<TrackerCPERecord>().get(cpeName,theEstimator);

    std::string cpeName = conf_.getParameter<std::string>("TTRHBuilder");
    cout << " Asking for the TTRHBuilder with name " << cpeName << endl;

    edm::ESHandle<TransientTrackingRecHitBuilder> theB;
    setup.get<TransientRecHitRecord>().get(cpeName, theB);
    auto& r = *theB;
    cout << " Got a " << typeid(r).name() << endl;
    //    cout <<" Strip CPE "<<theB->stripClusterParameterEstimator()<<endl;
    //    cout <<" Pixel CPE "<<theB->pixelClusterParameterEstimator()<<endl;
  }

private:
  edm::ParameterSet conf_;
};
