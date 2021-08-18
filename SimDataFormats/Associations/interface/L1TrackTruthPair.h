#ifndef SimDataFormats_Associations_L1TrackTruthPair_h
#define SimDataFormats_Associations_L1TrackTruthPair_h

#include "SimDataFormats/Associations/interface/TTTrackTruthPair.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>

using L1TrackTruthPair = TTTrackTruthPair<Ref_Phase2TrackerDigi_>;
using L1TrackTruthPairCollection = std::vector<L1TrackTruthPair>;

#endif
