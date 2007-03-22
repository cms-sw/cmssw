#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GlobalTrackingRegionProducer : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducer(const edm::ParameterSet& cfg) 
    : theRegionPSet(cfg.getParameter<edm::ParameterSet>("RegionPSet"))
  { }   

  virtual ~GlobalTrackingRegionProducer(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&, const edm::EventSetup&) const {
    std::vector<TrackingRegion* > result;
    result.push_back( new GlobalTrackingRegion(
        0.9, 0.2, 0.2, 0.0, true));
//        theConfig.getUntrackerParameter 
//        thePtMin, theOriginRadius, theOriginHalfLength, theOriginZPos, thePrecise) );
    return result;
  }

private:
  edm::ParameterSet theRegionPSet;
};

#endif 
