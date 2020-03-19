#ifndef TrackingAnalysis_EncodedTruthId_h
#define TrackingAnalysis_EncodedTruthId_h

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include <iosfwd>

class EncodedTruthId : public EncodedEventId {
  friend std::ostream &operator<<(std::ostream &os, const EncodedTruthId &id);

public:
  // Constructors
  EncodedTruthId();
  EncodedTruthId(EncodedEventId eid, int index);

  // Getters
  int index() const { return index_; }

  // Operators
  int operator==(const EncodedTruthId &id) const {
    if (EncodedEventId::operator==(id)) {
      return index_ == id.index_;
    } else {
      return EncodedEventId::operator==(id);
    }
  }

  int operator!=(const EncodedTruthId &id) const { return !(operator==(id)); }

  int operator<(const EncodedTruthId &id) const {
    if (EncodedEventId::operator==(id)) {
      return index_ < id.index_;
    } else {
      return (EncodedEventId::operator<(id));
    }
  }

private:
  int index_;
};

std::ostream &operator<<(std::ostream &os, EncodedTruthId &id);

#endif
