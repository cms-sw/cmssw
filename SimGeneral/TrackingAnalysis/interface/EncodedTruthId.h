#ifndef TrackingAnalysis_EncodedTruthId_h
#define TrackingAnalysis_EncodedTruthId_h

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class EncodedTruthId : public EncodedEventId {
 public:
  EncodedTruthId();
  EncodedTruthId(EncodedEventId eid, int index);
  int index() const {
    return index_;
  }
    
// Operators
  int operator==(const EncodedTruthId& id) const {
    return rawId() == id.rawId() && index_ == id.index_;
  }
  int operator!=(const EncodedTruthId& id) const {
    return rawId() != id.rawId() || index_ != id.index_;
  }
  int operator<( const EncodedTruthId& id) const {
    if (rawId() < id.rawId()) {
      return 1;
    } else {
      return index_ < id.index_;
    }    
  }
   
 private:  
  int index_;
};
  
#endif
