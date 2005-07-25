#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"
//#include "SimG4CMS/Tracker/interface/TouchableToHistory.h"

#include "G4VTouchable.hh"

int TrackerG4SimHitNumberingScheme::g4ToNumberingScheme(const G4VTouchable* pv){
  return 22;
}
TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(){
  //  ts = new TouchableToHistory;
}
TrackerG4SimHitNumberingScheme::~TrackerG4SimHitNumberingScheme(){
  //  if (ts) delete ts;
}
