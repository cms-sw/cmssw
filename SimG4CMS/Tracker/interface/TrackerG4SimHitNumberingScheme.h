#ifndef SimG4CMS_TrackerG4SimHitNumberingScheme_H
#define SimG4CMS_TrackerG4SimHitNumberingScheme_H

#include "SimG4CMS/Tracker/interface/TouchableToHistory.h"
#include "G4VTouchable.hh"

class DDCompactView;
class GeometricDet;

class TrackerG4SimHitNumberingScheme 
{
public:

  TrackerG4SimHitNumberingScheme(const DDCompactView&, const GeometricDet&);
  ~TrackerG4SimHitNumberingScheme();
    
  inline unsigned int g4ToNumberingScheme(const G4VTouchable* touch)
  { return ts->touchableToInt(touch); }

private:
  TouchableToHistory * ts;
};



#endif
