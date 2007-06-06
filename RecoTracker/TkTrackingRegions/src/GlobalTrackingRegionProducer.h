#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GlobalTrackingRegionProducer : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducer(const edm::ParameterSet& cfg) {

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
    theOriginZPos       = regionPSet.getParameter<double>("originZPos");
    thePrecise          = regionPSet.getParameter<bool>("precise");
  }

  virtual ~GlobalTrackingRegionProducer(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&, const edm::EventSetup&) const {
    std::vector<TrackingRegion* > result;
    result.push_back( new GlobalTrackingRegion(
        thePtMin, theOriginRadius, theOriginHalfLength, theOriginZPos, thePrecise) );
    return result;
  }

private:
  double thePtMin;
  double theOriginRadius;
  double theOriginHalfLength;
  double theOriginZPos;
  bool thePrecise;
};

#endif 
