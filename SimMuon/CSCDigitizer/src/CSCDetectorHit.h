#ifndef MU_END_DETECTOR_HIT_H
#define MU_END_DETECTOR_HIT_H

/** \class CSCDetectorHit
 *
 * A CSCDetectorHit can represent a hit either on a wire
 * or a strip in the early stages of the Endcap Muon CSC
 * digitization. See documentation for MEDigitizer subpackage of Muon.
 *
 * \author Rick Wilkinson
 */

#include <iosfwd>
class PSimHit;

class CSCDetectorHit
{
public:
  CSCDetectorHit(int element, float charge, float position, float time,
                   const PSimHit * hitp)
    : theElement(element), theCharge(charge),
      thePosition(position),   theTime(time), theHitp(hitp) {}

  int   getElement()  const {return theElement;}
  float getCharge()   const {return theCharge;}
  float getPosition() const {return thePosition;}
  float getTime()     const {return theTime;}
  const PSimHit * getSimHit() const {return theHitp;}

  friend std::ostream & operator<<(std::ostream &, const CSCDetectorHit &);
private:
  /// strip or wire number
  int   theElement;
  float theCharge;
  /// the position is along the element, with (0,0) the center of the chamber
  float thePosition; 
  /// start counting time at the beam crossing
  float theTime;
  /// theSimHit that created this hit
  const PSimHit * theHitp;
};

#endif

