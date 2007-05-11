#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GlobalTrackingRegionProducer : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducer(const edm::ParameterSet& cfg) 
    : thePtMin(cfg.getParameter<double>("ptMin")),
      theOriginRadius(cfg.getParameter<double>("originRadius")),
      theOriginHalfLength(cfg.getParameter<double>("originHalfLength")),
      theOriginZPos(cfg.getParameter<double>("originZPos")),
      thePrecise(cfg.getParameter<bool>("precise")) 
  { }   

  virtual ~GlobalTrackingRegionProducer(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&, const edm::EventSetup&) const {
    std::vector<TrackingRegion* > result;
    result.push_back( new GlobalTrackingRegion(
        thePtMin, theOriginRadius, theOriginHalfLength, theOriginZPos, thePrecise) );
    return result;
  }

private:
  double thePtMin, theOriginRadius, theOriginHalfLength, theOriginZPos;
  bool thePrecise;
};

#endif 
