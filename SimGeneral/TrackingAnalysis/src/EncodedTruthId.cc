#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

EncodedTruthId::EncodedTruthId() {}

EncodedTruthId::EncodedTruthId(EncodedEventId eid, int index):
    EncodedEventId(eid), index_(index) { }
