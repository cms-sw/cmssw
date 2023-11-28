#ifndef RecoTracker_PixelLowPtUtilities_plugins_PixelVertexProducerMedian_h
#define RecoTracker_PixelLowPtUtilities_plugins_PixelVertexProducerMedian_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class PixelVertexProducerMedian : public edm::global::EDProducer<> {
public:
  explicit PixelVertexProducerMedian(const edm::ParameterSet& ps);
  ~PixelVertexProducerMedian() override;
  void produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const override;

private:
  edm::ParameterSet theConfig;
  double thePtMin;
};
#endif
