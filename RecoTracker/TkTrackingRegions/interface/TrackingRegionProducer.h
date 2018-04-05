#ifndef RecoTracker_TkTrackingRegions_TrackingRegionProducer_H
#define RecoTracker_TkTrackingRegions_TrackingRegionProducer_H

#include <vector>
#include <memory>
class TrackingRegion;
namespace edm { class Event; class EventSetup; }

class TrackingRegionProducer {
public:
  virtual ~TrackingRegionProducer(){}
  virtual std::vector<std::unique_ptr<TrackingRegion> > 
      regions(const edm::Event& ev, const edm::EventSetup& es) const = 0;
};
#endif
