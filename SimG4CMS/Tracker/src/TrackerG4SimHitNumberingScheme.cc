#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4VTouchable.hh"
#include "G4TouchableHistory.hh"
#include "G4VSensitiveDetector.hh"

//#define DEBUG

TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(const DDCompactView& cpv, const GeometricDet& det)
    : alreadySet(false), myCompactView(&cpv), myGeomDet(&det) {}

TrackerG4SimHitNumberingScheme::~TrackerG4SimHitNumberingScheme() {}

void TrackerG4SimHitNumberingScheme::buildAll() {
  if (alreadySet)
    return;
  alreadySet = true;

  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator theNavigator;
  theNavigator.SetWorldVolume(theStdNavigator->GetWorldVolume());

  std::vector<const GeometricDet*> allSensitiveDets;
  myGeomDet->deepComponents(allSensitiveDets);
  edm::LogInfo("TrackerSimInfoNumbering")
      << " TouchableTo History: got " << allSensitiveDets.size() << " sensitive detectors from TrackerMapDDDtoID.";

  for (auto& theSD : allSensitiveDets) {
    DDTranslation const& t = theSD->translation();
    theNavigator.LocateGlobalPointAndSetup(G4ThreeVector(t.x(), t.y(), t.z()));
    G4TouchableHistory* hist = theNavigator.CreateTouchableHistory();
    TrackerG4SimHitNumberingScheme::Nav_Story st;
    touchToNavStory(hist, st);

    myMap[st] = Nav_type(theSD->navType().begin(), theSD->navType().end());
    myDirectMap[st] = theSD->geographicalID();

    LogDebug("TrackerSimDebugNumbering") << " INSERTING LV " << hist->GetVolume()->GetLogicalVolume()->GetName()
                                         << " SD: "
                                         << hist->GetVolume()->GetLogicalVolume()->GetSensitiveDetector()->GetName()
                                         << " Now size is " << myDirectMap.size();
    delete hist;
  }
  edm::LogInfo("TrackerSimInfoNumbering")
      << " TrackerG4SimHitNumberingScheme: mapped " << myDirectMap.size() << " detectors to Geant4.";

  if (myDirectMap.size() != allSensitiveDets.size()) {
    edm::LogError("TrackerSimInfoNumbering") << " ERROR: DDD sensitive detectors do not match Geant4 ones.";
    throw cms::Exception("TrackerG4SimHitNumberingScheme::buildAll")
        << " cannot resolve structure of tracking sensitive detectors";
  }
}

const DDFilteredView& TrackerG4SimHitNumberingScheme::getFilteredView(const G4VTouchable& t, DDFilteredView& f) {
  if (alreadySet == false) {
    buildAll();
  }
  TrackerG4SimHitNumberingScheme::Nav_Story st;
  touchToNavStory(&t, st);
  f.goTo(myMap[st]);
  return f;
}

TrackerG4SimHitNumberingScheme::Nav_type& TrackerG4SimHitNumberingScheme::getNavType(const G4VTouchable& t) {
  if (alreadySet == false) {
    buildAll();
  }
  TrackerG4SimHitNumberingScheme::Nav_Story st;
  touchToNavStory(&t, st);
  return myMap[st];
}

void TrackerG4SimHitNumberingScheme::getNavStory(DDFilteredView& i, TrackerG4SimHitNumberingScheme::Nav_Story& st) {
  if (alreadySet == false) {
    buildAll();
  }

  const DDTranslation& t = i.translation();

  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator theNavigator;
  theNavigator.SetWorldVolume(theStdNavigator->GetWorldVolume());

  theNavigator.LocateGlobalPointAndSetup(G4ThreeVector(t.x(), t.y(), t.z()));
  G4TouchableHistory* hist = theNavigator.CreateTouchableHistory();
  touchToNavStory(hist, st);
  delete hist;
}

void TrackerG4SimHitNumberingScheme::touchToNavStory(const G4VTouchable* v,
                                                     TrackerG4SimHitNumberingScheme::Nav_Story& st) {
#ifdef DEBUG
  std::vector<int> debugint;
  std::vector<std::string> debugstring;
#endif
  int levels = v->GetHistoryDepth();

  for (int k = 0; k <= levels; ++k) {
    if (v->GetVolume(k)->GetLogicalVolume()->GetName() != "TOBInactive") {
      st.push_back(
          std::pair<int, std::string>(v->GetVolume(k)->GetCopyNo(), v->GetVolume(k)->GetLogicalVolume()->GetName()));
#ifdef DEBUG
      debugint.push_back(v->GetVolume(k)->GetCopyNo());
      debugstring.push_back(v->GetVolume(k)->GetLogicalVolume()->GetName());
#endif
    }
  }
#ifdef DEBUG
  LogDebug("TrackerSimDebugNumbering") << " G4 TrackerG4SimHitNumberingScheme " << debugint;
  for (u_int32_t jj = 0; jj < debugstring.size(); jj++)
    LogDebug("TrackerSimDebugNumbering") << " " << debugstring[jj];
#endif
}

TrackerG4SimHitNumberingScheme::Nav_type& TrackerG4SimHitNumberingScheme::touchableToNavType(const G4VTouchable* v) {
  if (alreadySet == false) {
    buildAll();
  }
#ifdef DEBUG
  dumpG4VPV(v);
#endif
  TrackerG4SimHitNumberingScheme::Nav_Story st;
  touchToNavStory(v, st);
  return myMap[st];
}

unsigned int TrackerG4SimHitNumberingScheme::g4ToNumberingScheme(const G4VTouchable* v) {
  if (alreadySet == false) {
    buildAll();
  }
  TrackerG4SimHitNumberingScheme::Nav_Story st;
  touchToNavStory(v, st);

#ifdef DEBUG
  dumpG4VPV(v);
  LogDebug("TrackerSimDebugNumbering") << " Returning: " << myDirectMap[st];
#endif

  return myDirectMap[st];
}

void TrackerG4SimHitNumberingScheme::dumpG4VPV(const G4VTouchable* v) {
  int levels = v->GetHistoryDepth();

  LogDebug("TrackerSimDebugNumbering") << " NAME : " << v->GetVolume()->GetLogicalVolume()->GetName();
  for (int k = 0; k <= levels; k++) {
    LogDebug("TrackerSimInfoNumbering") << " Hist: " << v->GetVolume(k)->GetLogicalVolume()->GetName() << " Copy "
                                        << v->GetVolume(k)->GetCopyNo();
  }
}
