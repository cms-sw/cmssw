#ifndef Phase2TrackerDigiCommon_HH
#define Phase2TrackerDigiCommon_HH

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace phase2trackerdigi {
  int getOTLayerNumber(unsigned int& detid, const TrackerTopology* topo);
  int getPixelLayerNumber(unsigned int& detid, const TrackerTopology* topo);
}
#endif
