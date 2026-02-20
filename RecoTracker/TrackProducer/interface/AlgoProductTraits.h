#ifndef RecoTrackerTrackProducerAlgoProductTraits_H
#define RecoTrackerTrackProducerAlgoProductTraits_H

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFwd.h"
#include "DataFormats/Common/interface/View.h"
#include <vector>

template <class T>
class AlgoProductTraits {
public:
  using TrackCollection = std::vector<T>;
  using TrackView = edm::View<T>;
  struct AlgoProduct {
    Trajectory* trajectory;
    T* track;
    PropagationDirection pDir;
    int indexInput;
  };
  using AlgoProductCollection = std::vector<AlgoProduct>;
};
#endif  //AlgoProductTraits_H
