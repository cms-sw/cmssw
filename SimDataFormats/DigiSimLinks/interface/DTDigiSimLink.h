#ifndef DigiSimLinks_DTDigiSimLink_h
#define DigiSimLinks_DTDigiSimLink_h

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class DTDigiSimLink {
public:
  typedef uint32_t ChannelType;

  // Construct from the wire number and the digi number (this identifies
  // uniquely multiple digis on the same wire), the TDC counts, the SimTrack Id and the EncodedEvent Id.
  // Base is related to the tdc unit (32 = Phase 1; 30 = Phase 2)
  explicit DTDigiSimLink(int wireNr, int digiNr, int nTDC, unsigned int trackId, EncodedEventId evId, int base = 32);

  // Construct from the wire number and the digi number (this identifies
  // uniquely multiple digis on the same wire), the time (ns), the SimTrack Id and the EncodedEvent Id.
  // time is converted in TDC counts (1 TDC = 25./32. ns)
  explicit DTDigiSimLink(
      int wireNr, int digiNr, double tdrift, unsigned int trackId, EncodedEventId evId, int base = 32);

  // Default constructor.
  DTDigiSimLink();

  // The channel identifier and the digi number packed together
  ChannelType channel() const;

  // Return wire number
  int wire() const;

  // Identifies different digis within the same cell
  int number() const;

  // Get raw TDC count
  uint32_t countsTDC() const;

  // Get time in ns
  double time() const;

  // Return the SimTrack Id
  unsigned int SimTrackId() const;

  // Return the Encoded Event Id
  EncodedEventId eventId() const;

  // Used to repack the channel number to an int
  struct ChannelPacking {
    uint16_t wi;
    uint16_t num;
  };

private:
  // The value of one TDC count in ns
  static const double reso;

private:
  uint16_t theWire;        // wire number
  uint8_t theDigiNumber;   // counter for digis in the same cell
  uint8_t theTDCBase;      // TDC base (counts per BX; 32 in Ph1 or 30 in Ph2)
  int32_t theCounts;       // TDC count, in units given by 1/theTDCBase
  uint32_t theSimTrackId;  // identifier of the SimTrack that produced the digi
  EncodedEventId theEventId;
};

#include <iostream>
#include <cstdint>
inline std::ostream& operator<<(std::ostream& o, const DTDigiSimLink& digisimlink) {
  return o << "wire:" << digisimlink.wire() << " digi:" << digisimlink.number() << " time:" << digisimlink.time()
           << " SimTrack:" << digisimlink.SimTrackId() << " eventId:" << digisimlink.eventId().rawId();
}

#endif
