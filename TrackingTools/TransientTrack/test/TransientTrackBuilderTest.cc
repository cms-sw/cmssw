#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"


#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"


#include <iostream>
#include <string>

using namespace edm;

class TransientTrackBuilderTest : public edm::EDAnalyzer {
 public:
  TransientTrackBuilderTest(const edm::ParameterSet& pset) {conf_ = pset;}

  ~TransientTrackBuilderTest(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){
        using namespace std;

//     std::string cpeName = conf_.getParameter<std::string>("PixelCPE");   
//     cout <<" Asking for the CPE with name "<<cpeName<<endl;

//     edm::ESHandle<PixelClusterParameterEstimator> theEstimator;
//     setup.get<TrackerCPERecord>().get(cpeName,theEstimator);



    std::string cpeName("TransientTrackBuilder");   
    cout <<" Asking for the TransientTrackBuilder with name "<<cpeName<<endl;

    edm::ESHandle<TransientTrackBuilder> theB;
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    
    cout <<" Got a "<<typeid(*theB).name()<<endl;
    cout << "Field at origin (in Testla): "<< (*theB).field()->inTesla(GlobalPoint(0.,0.,0.))<<endl;
    
  }
private:
  edm::ParameterSet conf_;
};

