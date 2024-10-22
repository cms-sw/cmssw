#ifndef Tracking_TempMeasurements_H
#define Tracking_TempMeasurements_H
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>
#include <algorithm>

namespace tracking {
  // the return type from a MeasurementDet
  struct TempMeasurements {
    typedef TrackingRecHit::ConstRecHitContainer RecHitContainer;
    typedef TrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
    typedef std::vector<float> Distances;

    void clear() {
      hits.clear();
      distances.clear();
    }
    bool empty() const { return hits.empty(); }
    std::size_t size() const { return hits.size(); }

    inline void sortIndex(int* index) const {
      float const* d = &distances.front();  // do not trust capture
      for (std::size_t i = 0; i != size(); ++i) {
        index[i] = i;
        std::push_heap(index, index + i + 1, [d](int j, int k) { return d[j] < d[k]; });
      }
      std::make_heap(index, index + size(), [d](int j, int k) { return d[j] < d[k]; });
    }

    void add(ConstRecHitPointer const& h, float d) {
      hits.push_back(h);
      distances.push_back(d);
    }
    void add(ConstRecHitPointer&& h, float d) {
      hits.push_back(std::move(h));
      distances.push_back(d);
    }

    RecHitContainer hits;
    Distances distances;
  };
}  // namespace tracking

#endif  // Tracking_TempMeas_H
