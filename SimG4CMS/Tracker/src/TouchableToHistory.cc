#include "SimG4CMS/Tracker/interface/TouchableToHistory.h"
#include "Geometry/TrackerBaseAlgo/interface/TrackerMapDDDtoID.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <iomanip>
#include <fstream>

#include "G4TransportationManager.hh" 
#include "G4Navigator.hh" 
#include "G4VTouchable.hh"
#include "G4TouchableHistory.hh"

//#define DEBUG
//#define DEBUGMUCH
void TouchableToHistory::buildAll(){
  if (alreadySet == true) return;
  alreadySet = true;

  
  std::vector<nav_type> allSensitiveDets = TrackerMapDDDtoID::instance().allNavTypes();
  std::cout <<" TouchableTo History: got "<<allSensitiveDets.size()<<" sensitive detectors from TrackerMapDDDtoID."<<std::endl;
  DDCompactView cv;
  DDExpandedView view(cv);
  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator* theNavigator = new G4Navigator();
  theNavigator->SetWorldVolume(theStdNavigator->GetWorldVolume());

  for (std::vector<nav_type>::iterator it = allSensitiveDets.begin();  it != allSensitiveDets.end(); it++){
    view.goTo(*it);
    DDTranslation t =view.translation(); 
    theNavigator->LocateGlobalPointAndSetup(G4ThreeVector(t.x(),t.y(),t.z()));
    G4TouchableHistory * hist = theNavigator->CreateTouchableHistory(); 
    TouchableToHistory::Nav_Story st =  touchableToNavStory(hist);
    
#ifdef DEBUG
    int oldsize = myDirectMap.size();
#endif
    myMap[st] = *it;
    myDirectMap[st] = TrackerMapDDDtoID::instance().id(*it);
#ifdef DEBUG
    std::cout << " INSERTING "<<view.logicalPart().name()<<" "<<t<<" "<<*it<<" "<<hist->GetVolume()->GetLogicalVolume()->GetName();
    std::cout <<" Sensitive: "<<hist->GetVolume()->GetLogicalVolume()->GetSensitiveDetector()<<std::endl;
    std::cout <<"Now size is "<<myDirectMap.size()<<std::endl;
    if (oldsize == myDirectMap.size())
      std::cout<< "Touchable to History Error!!!!"<<std::endl;
    dumpG4VPV(hist);
    std::cout <<"-------------"<<std::endl;
#endif
    delete hist;
  }
  std::cout <<" TouchableToHistory: mapped "<<myDirectMap.size()<<" detectors to G4."<<std::endl;
  if (myDirectMap.size() != allSensitiveDets.size()){
    std::cout <<" ERROR: DDD sensitive detectors do not match Geant4 ones."<<std::endl;
    abort();
  }
  TrackerMapDDDtoID::instance().clear();

  delete theNavigator;

}

DDFilteredView& TouchableToHistory::getFilteredView(const G4VTouchable& t, DDFilteredView& f){
  if (alreadySet == false) 
    buildAll();
  f.goTo(myMap[touchableToNavStory(&t)]);
  return f;
}
TouchableToHistory::nav_type TouchableToHistory::getNavType(const G4VTouchable& t){
  if (alreadySet == false) 
    std::cerr<<" NOT READY "<<std::endl;
  return myMap[touchableToNavStory(&t)];
}

TouchableToHistory::Nav_Story TouchableToHistory::getNavStory(DDFilteredView& i){
  if (alreadySet == false) buildAll();
  DDTranslation t = i.translation();
  //  G4Navigator theNavigator;
  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator* theNavigator = new G4Navigator();
  theNavigator->SetWorldVolume(theStdNavigator->GetWorldVolume());

  theNavigator->LocateGlobalPointAndSetup(G4ThreeVector(t.x(),t.y(),t.z()));
  G4TouchableHistory* hist = theNavigator->CreateTouchableHistory(); 
  TouchableToHistory::Nav_Story temp = touchableToNavStory(hist);
  delete hist;
  delete theNavigator;
  return (temp);
}

TouchableToHistory::Nav_Story TouchableToHistory::touchableToNavStory(const G4VTouchable *v) {
  Nav_Story temp;
  int levels = v->GetHistoryDepth();
  
  for (int k=0; k<=levels; k++){
    if (v->GetVolume(k)->GetLogicalVolume()->GetName() != "TOBInactive") 
      temp.push_back(
		     std::pair<int,std::string>
		     (v->GetVolume(k)->GetCopyNo(),
		      v->GetVolume(k)->GetLogicalVolume()->GetName()));
  }
#ifdef DEBUG
  std::cout << " TouchableToHistory::touchableToNavStory returning "<<temp<<std::endl;
#endif
  return temp;
}

TouchableToHistory::nav_type TouchableToHistory::touchableToNavType(const G4VTouchable* v) {
  if (alreadySet == false) 
    buildAll();
#ifdef DEBUGMUCH
  std::cout <<" NAME : "<<v->GetVolume()->GetLogicalVolume()->GetName()<<std::endl;
    dumpG4VPV(v);
#endif
  return   myMap[touchableToNavStory(v)];
}
int TouchableToHistory::touchableToInt(const G4VTouchable* v) {
  if (alreadySet == false) 
    buildAll();
#ifdef DEBUG
  std::cout <<" NAME : "<<v->GetVolume()->GetLogicalVolume()->GetName()<<std::endl;
    dumpG4VPV(v);
    std::cout <<" Returning: "<< myDirectMap[touchableToNavStory(v)]<<std::endl;
#endif
  return   myDirectMap[touchableToNavStory(v)];
}

void TouchableToHistory::dumpG4VPV(const G4VTouchable* v){
  int levels = v->GetHistoryDepth();
  
  for (int k=0; k<=levels; k++){
    std::cout <<" Hist: "<< v->GetVolume(k)->GetLogicalVolume()->GetName()<<
      " Copy "<< v->GetVolume(k)->GetCopyNo()<<std::endl;
  }
}

