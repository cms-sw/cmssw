#ifdef SLHC_DT_TRK_DFENABLE

#include "SimDataFormats/SLHC/interface/TrackTriggerPrimitive.h"

// default ctor
TrackTriggerPrimitive::TrackTriggerPrimitive() :
  hits_(0)
{ }

// ctor from one hit
TrackTriggerPrimitive::TrackTriggerPrimitive(const TrackTriggerHit& h)
{
  hits_.push_back(h);
}

// ctor from two hits
TrackTriggerPrimitive::TrackTriggerPrimitive(const TrackTriggerHit& h1, const TrackTriggerHit& h2)
{
  hits_.push_back(h1);
  hits_.push_back(h2);
}

// dtor
TrackTriggerPrimitive::~TrackTriggerPrimitive() { }

/// get all hits
std::vector< TrackTriggerHit > TrackTriggerPrimitive::getHits() const {
  return hits_;
}

/// get number of hits
unsigned TrackTriggerPrimitive::nHits() const {
  return hits_.size();
}

/// get single hit
TrackTriggerHit TrackTriggerPrimitive::hit(unsigned i) const {
  return hits_.at(i);
}

std::ostream& operator << (std::ostream& os, const TrackTriggerPrimitive& tp) {
  os << "TrackTriggerPrimitive : hit rows = ";
  for (unsigned i=0; i<tp.nHits(); ++i) {
    os << tp.hit(i).row() << " ";
  }
  return os;
}

#endif
