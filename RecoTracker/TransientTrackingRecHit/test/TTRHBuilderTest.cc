// user includes
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include <memory>
#include <iostream>
#include <string>

using namespace edm;

class TTRHBuilderTest : public edm::global::EDAnalyzer<> {
public:
  TTRHBuilderTest(const edm::ParameterSet& pset)
      : cpeName_(pset.getParameter<std::string>("TTRHBuilder")),
        ttrhToken_(esConsumes(edm::ESInputTag("", cpeName_))) {}
  //pixelCPEToken_(esConsumes(edm::ESInputTag("",cpeName_))){}
  ~TTRHBuilderTest() = default;

  virtual void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
    using namespace std;

    // edm::LogPrint("TTRBuilderTest") <<" Asking for the CPE with name "<<cpeName<<endl;
    // const PixelClusterParameterEstimator* theEstimator = &iSetup.getData(pixelCPEToken_);

    edm::LogPrint("TTRHBuilderTest") << " Asking for the TTRHBuilder with name " << cpeName_ << endl;
    const TransientTrackingRecHitBuilder* r = &iSetup.getData(ttrhToken_);
    edm::LogPrint("TTRBuilderTest") << " Got a " << typeid(r).name() << endl;
  }

private:
  const std::string cpeName_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhToken_;
  //const edm::ESGetToken<PixelClusterParameterEstimator,TrackerCPERecord> pixelCPEToken_;
};
