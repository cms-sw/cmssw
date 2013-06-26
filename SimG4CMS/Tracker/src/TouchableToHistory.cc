#include "SimG4CMS/Tracker/interface/TouchableToHistory.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"


#include "G4TransportationManager.hh" 
#include "G4Navigator.hh" 
#include "G4VTouchable.hh"
#include "G4TouchableHistory.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DEBUG

void TouchableToHistory::buildAll(){
  if (alreadySet) return;
  alreadySet = true;

  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator theNavigator;
  theNavigator.SetWorldVolume(theStdNavigator->GetWorldVolume());


  std::vector<const GeometricDet*> allSensitiveDets;
  myGeomDet->deepComponents(allSensitiveDets);
  edm::LogInfo("TrackerSimInfoNumbering")<<" TouchableTo History: got "<<allSensitiveDets.size()<<" sensitive detectors from TrackerMapDDDtoID.";

  for ( std::vector<const GeometricDet*>::const_iterator it = allSensitiveDets.begin(); 
	it != allSensitiveDets.end(); 
	++it)
    {
      DDTranslation const & t = (*it)->translation(); 
      theNavigator.LocateGlobalPointAndSetup(G4ThreeVector(t.x(),t.y(),t.z()));
      G4TouchableHistory * hist = theNavigator.CreateTouchableHistory(); 
      TouchableToHistory::Nav_Story st =  touchableToNavStory(hist);

#ifdef DEBUG    
    u_int32_t oldsize = myDirectMap.size();
#endif

    myMap[st] = nav_type((*it)->navType().begin(),(*it)->navType().end());
    myDirectMap[st] = (*it)->geographicalID();

    /*
#ifdef DEBUG    
    LogDebug("TrackerSimDebugNumbering")<< " INSERTING "<<view.logicalPart().name()<<" "<<t<<" "<<hist->GetVolume()->GetLogicalVolume()->GetName();
    LogDebug("TrackerSimDebugNumbering")<<" Sensitive: "<<hist->GetVolume()->GetLogicalVolume()->GetSensitiveDetector()<<std::endl;
    LogDebug("TrackerSimDebugNumbering")<<"Now size is "<<myDirectMap.size()<<std::endl;
    if (oldsize == myDirectMap.size())
      edm::LogError("TrackerSimInfoNumbering")<< "Touchable to History Error!!!!";
    dumpG4VPV(hist);
#endif
    */
    delete hist;

  }
  edm::LogInfo("TrackerSimInfoNumbering")<<" TouchableToHistory: mapped "<<myDirectMap.size()<<" detectors to G4.";

  if (myDirectMap.size() != allSensitiveDets.size()){
    edm::LogError("TrackerSimInfoNumbering")<<" ERROR: DDD sensitive detectors do not match Geant4 ones.";
    //FIXME use throw
    abort();
  }


}

DDFilteredView& TouchableToHistory::getFilteredView(const G4VTouchable& t, DDFilteredView& f){
  if (alreadySet == false) 
    buildAll();
  f.goTo(myMap[touchableToNavStory(&t)]);
  return f;
}
TouchableToHistory::nav_type TouchableToHistory::getNavType(const G4VTouchable& t){
  if (alreadySet == false) 
    edm::LogError("TrackerSimInfoNumbering")<<" NOT READY ";
  return myMap[touchableToNavStory(&t)];
}

TouchableToHistory::Nav_Story TouchableToHistory::getNavStory(DDFilteredView& i){
  if (alreadySet == false) buildAll();
  DDTranslation t = i.translation();

  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator  theNavigator;
  theNavigator.SetWorldVolume(theStdNavigator->GetWorldVolume());

  theNavigator.LocateGlobalPointAndSetup(G4ThreeVector(t.x(),t.y(),t.z()));
  G4TouchableHistory* hist = theNavigator.CreateTouchableHistory(); 
  TouchableToHistory::Nav_Story temp = touchableToNavStory(hist);
  delete hist;
  return (temp);
}

TouchableToHistory::Nav_Story TouchableToHistory::touchableToNavStory(const G4VTouchable *v) {
  static G4String tobinactive("TOBInactive");
  Nav_Story temp;
#ifdef DEBUG    
  std::vector<int> debugint;
  std::vector<std::string> debugstring;
#endif
  int levels = v->GetHistoryDepth();
  
  for (int k=0; k<=levels; k++){
    if (v->GetVolume(k)->GetLogicalVolume()->GetName() != tobinactive) {
      temp.push_back(
		     std::pair<int,std::string>
		     (v->GetVolume(k)->GetCopyNo(),
		      v->GetVolume(k)->GetLogicalVolume()->GetName()));
#ifdef DEBUG    
      debugint.push_back(v->GetVolume(k)->GetCopyNo());
      debugstring.push_back(v->GetVolume(k)->GetLogicalVolume()->GetName());
#endif
    }
  }
#ifdef DEBUG    
  // LogDebug("TrackerSimDebugNumbering")<<" G4 TouchableToHistory "<< debugint;
  for(u_int32_t jj=0;jj<debugstring.size();jj++)LogDebug("TrackerSimDebugNumbering")<<" "<<debugstring[jj];
#endif
  return temp;
}

TouchableToHistory::nav_type TouchableToHistory::touchableToNavType(const G4VTouchable* v) {
  if (alreadySet == false) 
    buildAll();

  dumpG4VPV(v);

  return   myMap[touchableToNavStory(v)];
}

int TouchableToHistory::touchableToInt(const G4VTouchable* v) {
  if (alreadySet == false) 
    buildAll();

  dumpG4VPV(v);

  LogDebug("TrackerSimDebugNumbering")<<" Returning: "<< myDirectMap[touchableToNavStory(v)]<<std::endl;

  return   myDirectMap[touchableToNavStory(v)];
}

void TouchableToHistory::dumpG4VPV(const G4VTouchable* v){
  int levels = v->GetHistoryDepth();
  
  LogDebug("TrackerSimDebugNumbering")<<" NAME : "<<v->GetVolume()->GetLogicalVolume()->GetName();
  for (int k=0; k<=levels; k++){
    LogDebug("TrackerSimInfoNumbering") <<" Hist: "<< v->GetVolume(k)->GetLogicalVolume()->GetName()<<
      " Copy "<< v->GetVolume(k)->GetCopyNo();
  }
}

