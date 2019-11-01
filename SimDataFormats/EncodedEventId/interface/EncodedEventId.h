#ifndef SimDataFormats_EncodedEventId_H
#define SimDataFormats_EncodedEventId_H 1

#include <TMath.h>
#include <ostream>
#include <cstdint>

/** \class EncodedEventId

*/
class EncodedEventId {
public:
  /// Create an empty or null id (also for persistence)
  EncodedEventId();
  /// Create an id from a raw number
  explicit EncodedEventId(uint32_t id);
  /// Create an id, filling the bunch crossing and event infomrations
  EncodedEventId(int bunchX, int event) {
    id_ = TMath::Abs(bunchX) << bunchXStartBit_ | event;
    if (bunchX < 0)
      id_ = id_ | bunchNegMask_;
  }

  /// get the detector field from this detid
  int bunchCrossing() const {
    int bcr = int((id_ >> bunchXStartBit_) & 0x7FFF);
    return id_ & bunchNegMask_ ? -bcr : bcr;
  }
  /// get the contents of the subdetector field (should be protected?)
  int event() const { return int(id_ & 0xFFFF); }
  uint32_t operator()() { return id_; }
  /// get the raw id
  uint32_t rawId() const { return id_; }
  /// equality
  int operator==(const EncodedEventId& id) const { return id_ == id.id_; }
  /// inequality
  int operator!=(const EncodedEventId& id) const { return id_ != id.id_; }
  /// comparison
  int operator<(const EncodedEventId& id) const { return id_ < id.id_; }

private:
  static const unsigned int bunchXStartBit_ = 16;
  static const unsigned int eventStartBit_ = 0;
  static const unsigned int bunchXMask_ = 0x10;
  static const unsigned int bunchNegMask_ = 0x80000000;
  static const unsigned int eventMask_ = 0x10;

protected:
  uint32_t id_;
};
#endif
