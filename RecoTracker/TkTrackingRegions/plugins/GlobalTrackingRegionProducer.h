#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducer_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class GlobalTrackingRegionProducer : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducer(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector && iC) { 

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
    double xPos         = regionPSet.getParameter<double>("originXPos");
    double yPos         = regionPSet.getParameter<double>("originYPos");
    double zPos         = regionPSet.getParameter<double>("originZPos");
    thePrecise          = regionPSet.getParameter<bool>("precise"); 
    theOrigin = GlobalPoint(xPos,yPos,zPos);
  }   

  virtual ~GlobalTrackingRegionProducer(){}

  virtual std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event&, const edm::EventSetup&) const override {
    std::vector<std::unique_ptr<TrackingRegion> > result;
    result.push_back( 
        std::make_unique<GlobalTrackingRegion>( thePtMin, theOrigin, theOriginRadius, theOriginHalfLength, thePrecise) );
    return result;
  }

private:
  double thePtMin; 
  GlobalPoint theOrigin;
  double theOriginRadius; 
  double theOriginHalfLength; 
  bool thePrecise;
};

#endif 
