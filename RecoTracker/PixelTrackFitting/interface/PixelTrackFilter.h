#ifndef RecoTracker_PixelTrackFitting_PixelTrackFilter_h
#define RecoTracker_PixelTrackFitting_PixelTrackFilter_h

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilterBase.h"

#include <memory>

class PixelTrackFilter {
public:
  PixelTrackFilter() {}
  explicit PixelTrackFilter(std::unique_ptr<PixelTrackFilterBase> filter) : filter_(std::move(filter)) {}

  void swap(PixelTrackFilter& o) { std::swap(filter_, o.filter_); }

  bool operator()(const reco::Track* track, const PixelTrackFilterBase::Hits& hits) const {
    return (*filter_)(track, hits);
  }

private:
  std::unique_ptr<PixelTrackFilterBase> filter_;
};

#endif
