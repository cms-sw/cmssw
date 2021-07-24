#ifndef GEMOBJECTS_GEMDIGISIMLINK_H
#define GEMOBJECTS_GEMDIGISIMLINK_H

#include <map>
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

class GEMDigiSimLink {
public:
  GEMDigiSimLink(uint16_t strip, int8_t  bx, uint32_t detUnitId, 
                 uint32_t trackId, EncodedEventId eventId)
  {
    strip_ = strip;
    bx_ = bx;
    detUnitId_ = detUnitId;
    trackId_ = trackId;
    eventId_ = eventId;
  }

  GEMDigiSimLink() { }
  ~GEMDigiSimLink() { }

  unsigned int getStrip() const { return strip_; }
  int getBx() const { return bx_; }
  unsigned int getDetUnitId() const { return detUnitId_; }
  unsigned int getTrackId() const { return trackId_; }
  EncodedEventId getEventId() const { return eventId_; }

  inline bool operator<(const GEMDigiSimLink& other) const { return getStrip() < other.getStrip(); }

private:
  uint16_t strip_;
  int8_t  bx_;
  uint32_t detUnitId_;
  uint32_t trackId_;
  EncodedEventId eventId_;
};
#endif
