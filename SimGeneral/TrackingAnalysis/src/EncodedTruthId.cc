#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

EncodedTruthId::EncodedTruthId() {}

EncodedTruthId::EncodedTruthId(EncodedEventId eid, int index) : EncodedEventId(eid), index_(index) {}

std::ostream &operator<<(std::ostream &os, EncodedTruthId &id) {
  return os << "(" << id.bunchCrossing() << "," << id.event() << "," << id.index() << ")";
}
