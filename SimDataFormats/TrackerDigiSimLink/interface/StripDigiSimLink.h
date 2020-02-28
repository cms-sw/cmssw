#ifndef SimDataFormats_TrackerDigiSimLink_StripDigiSimLink_h
#define SimDataFormats_TrackerDigiSimLink_StripDigiSimLink_h

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include <cstdint>

class StripDigiSimLink {
public:
  enum { LowTof, HighTof };

  StripDigiSimLink(
      unsigned int ch, unsigned int tkId, unsigned int counter, unsigned int tofBin, EncodedEventId e, float a)
      : chan(ch),
        simTkId(tkId),
        CFpos(tofBin == LowTof ? counter & 0x7FFFFFFF : (counter & 0x7FFFFFFF) | 0x80000000),
        eId(e),
        fract(a) {
    ;
  }

  StripDigiSimLink(unsigned int ch, unsigned int tkId, unsigned int counter, EncodedEventId e, float a)
      : chan(ch), simTkId(tkId), CFpos(counter & 0x7FFFFFFF), eId(e), fract(a) {
    ;
  }

  StripDigiSimLink(unsigned int ch, unsigned int tkId, EncodedEventId e, float a)
      : chan(ch), simTkId(tkId), CFpos(0), eId(e), fract(a) {
    ;
  }

  StripDigiSimLink() : chan(0), simTkId(0), CFpos(0), eId(0), fract(0) { ; }

  ~StripDigiSimLink() { ; }

  unsigned int channel() const { return chan; }
  unsigned int SimTrackId() const { return simTkId; }
  unsigned int CFposition() const { return CFpos & 0x7FFFFFFF; }
  unsigned int TofBin() const { return (CFpos & 0x80000000) == 0 ? LowTof : HighTof; }
  EncodedEventId eventId() const { return eId; }
  float fraction() const { return fract; }

  inline bool operator<(const StripDigiSimLink& other) const { return channel() < other.channel(); }

private:
  unsigned int chan;
  unsigned int simTkId;
  uint32_t CFpos;  // position of the PSimHit in the CrossingFrame vector
                   // for the subdetector collection; bit 31 set if from the HighTof collection
  EncodedEventId eId;
  float fract;
};
#endif
