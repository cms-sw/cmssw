#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include <utility>

namespace io_v1 {

  CrossingFramePlaybackInfoNew::CrossingFramePlaybackInfoNew(int minBunch, int maxBunch, unsigned int maxNbSources)
      : maxNbSources_(maxNbSources),
        nBcrossings_(maxBunch - minBunch + 1),
        sizes_(maxNbSources_ * nBcrossings_, 0U),
        eventInfo_(),
        minBunch_(minBunch) {}

}  // namespace io_v1
