#ifdef SLHC_DT_TRK_DFENABLE

#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"

///null hit
TrackTriggerHit::TrackTriggerHit() :
  row_(0),
  col_(0)
{ }

// onstruct from id
TrackTriggerHit::TrackTriggerHit(unsigned row, unsigned col) :
  row_(row),
  col_(col)
{ }

/// copy ctor
TrackTriggerHit::TrackTriggerHit(const TrackTriggerHit& h) :
  row_(h.row()),
  col_(h.column())

{ }

/// dtor
TrackTriggerHit::~TrackTriggerHit() { }

/// comparison
bool operator < ( const TrackTriggerHit& a, const TrackTriggerHit& b )
{
	if ( a.row() < b.row() ) return true;
	if ( a.row() == b.row() && a.column() < b.column() ) return true;
	return false;
}

/// pretty print
std::ostream& operator << (std::ostream& os, const TrackTriggerHit& hit) {
  os << "TrackTriggerHit : row=" << hit.row();
  os << " col=" << hit.column();
  return os;
}

#endif
