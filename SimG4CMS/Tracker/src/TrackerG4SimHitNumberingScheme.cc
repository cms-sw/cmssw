#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"
#include "SimG4CMS/Tracker/interface/TouchableToHistory.h"

#include "G4VTouchable.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4TouchableHistory.hh"

unsigned int TrackerG4SimHitNumberingScheme::g4ToNumberingScheme(const G4VTouchable* pv){
  return ts->touchableToInt(pv);
}
TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(const DDCompactView& cpv,
   const GeometricDet& det ){
   ts = new TouchableToHistory(cpv,det);
}
void TrackerG4SimHitNumberingScheme::clear(){
  if (ts) delete ts;
}
