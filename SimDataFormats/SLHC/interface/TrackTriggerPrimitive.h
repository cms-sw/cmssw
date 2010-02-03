#ifndef L1TRACKERTRIGPRIM
#define L1TRACKERTRIGPRIM

/** \class L1TrackTrigPrim
 *
 *  Store an SLHC track trigger primitive/stub
 *  Implemented as a vector of hits
 *
 *  $Date: 2009/05/18 16:23:33 $
 *  $Revision: 1.1 $
 *  \author Jim Brooke 
*/

#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"

#include <vector>


class TrackTriggerPrimitive {

 public:

  /// default ctor
  TrackTriggerPrimitive();

  // construct from a hit
  TrackTriggerPrimitive(const TrackTriggerHit& h);

  // construct from a pair of hits
  TrackTriggerPrimitive(const TrackTriggerHit& h1, const TrackTriggerHit& h2);

  /// dtor
  ~TrackTriggerPrimitive();


  /// get all hits
  std::vector< TrackTriggerHit > getHits() const;

  // get number of hits
  unsigned nHits() const;

  /// get single hit
  TrackTriggerHit hit(unsigned i) const;

 private:

  // vector of hits
  std::vector< TrackTriggerHit > hits_;

};

std::ostream& operator << (std::ostream& os, const TrackTriggerPrimitive& tp);

#endif

