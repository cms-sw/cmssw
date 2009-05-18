#ifndef TRACKTRIGGERHIT
#define TRACKTRIGGERHIT

/** \class TrackTriggerHit
 *  class to represent a tracker hit for the L1 trigger
 *
 *  Store in a DetSetVector to provide a DetId
 *
 *  $Date: 2009/02/17 12:25:20 $
 *  $Revision: 1.4 $
 *  \author Jim Brooke 
*/

#include <iostream>

class TrackTriggerHit {
  
 public:
  
  /// null hit
  TrackTriggerHit();

  /// construct hit from id
  TrackTriggerHit(unsigned row, unsigned col);

  /// copy constructor 
  TrackTriggerHit(const TrackTriggerHit& h);
   
  /// dtor
  ~TrackTriggerHit();

  /// get row
  unsigned row() const { return row_; }

  /// get column
  unsigned column() const { return col_; }

/*  bool operator < (const TrackTriggerHit& other)
  {
    if ( row() < other.row() ) return true;
    if ( row() == other.row() && column() < other.column() ) return true;
    return false;
  }*/

 private:

  unsigned row_;
  unsigned col_;

};

bool operator < ( const TrackTriggerHit& a, const TrackTriggerHit& b );
std::ostream& operator << (std::ostream& os, const TrackTriggerHit& hit);


#endif
