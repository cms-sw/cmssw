#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"
#include "Geometry/TrackerBaseAlgo/interface/TrackerMapDDDtoID.h"

#include "G4VTouchable.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4TouchableHistory.hh"

unsigned int TrackerG4SimHitNumberingScheme::g4ToNumberingScheme(const G4VTouchable* pv){
  nav_type temp;
  int levels = pv->GetHistoryDepth();
  
  for (int k=0; k<=levels; k++){
    G4VPhysicalVolume* vol = pv->GetVolume(k);
    int copyno=vol->GetCopyNo();
    temp.push_back(copyno);
  }
  
  return TrackerMapDDDtoID::instance().id(temp);
}
TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(){
}
TrackerG4SimHitNumberingScheme::~TrackerG4SimHitNumberingScheme(){
}
