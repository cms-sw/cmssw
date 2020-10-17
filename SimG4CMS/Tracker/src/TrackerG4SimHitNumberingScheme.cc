#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4VTouchable.hh"
#include "G4TouchableHistory.hh"
#include "G4VSensitiveDetector.hh"

//#define DEBUG

TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(const GeometricDet& det)
    : alreadySet_(false), geomDet_(&det) {}

void TrackerG4SimHitNumberingScheme::buildAll() {
  if (alreadySet_)
    return;
  alreadySet_ = true;

  G4Navigator* theStdNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4Navigator theNavigator;
  theNavigator.SetWorldVolume(theStdNavigator->GetWorldVolume());

  std::vector<const GeometricDet*> allSensitiveDets;
  geomDet_->deepComponents(allSensitiveDets);
  edm::LogVerbatim("TrackerSimInfoNumbering")
      << " TouchableTo History: got " << allSensitiveDets.size() << " sensitive detectors from GeometricDet.";

  for (auto& theSD : allSensitiveDets) {
    auto const& t = theSD->translation();
    theNavigator.LocateGlobalPointAndSetup(G4ThreeVector(t.x(), t.y(), t.z()));
    G4TouchableHistory* hist = theNavigator.CreateTouchableHistory();
    TrackerG4SimHitNumberingScheme::Nav_Story st;
    touchToNavStory(hist, st);

    directMap_[st] = theSD->geographicalId();

    LogDebug("TrackerSimDebugNumbering") << " INSERTING LV " << hist->GetVolume()->GetLogicalVolume()->GetName()
                                         << " SD: "
                                         << hist->GetVolume()->GetLogicalVolume()->GetSensitiveDetector()->GetName()
                                         << " Now size is " << directMap_.size();
    delete hist;
  }
  edm::LogVerbatim("TrackerSimInfoNumbering")
      << " TrackerG4SimHitNumberingScheme: mapped " << directMap_.size() << " detectors to Geant4.";

  if (directMap_.size() != allSensitiveDets.size()) {
    edm::LogError("TrackerSimInfoNumbering") << " ERROR: GeomDet sensitive detectors do not match Geant4 ones.";
    throw cms::Exception("TrackerG4SimHitNumberingScheme::buildAll")
        << " cannot resolve structure of tracking sensitive detectors";
  }
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
      st.emplace_back(
          std::pair<int, std::string>(v->GetVolume(k)->GetCopyNo(), v->GetVolume(k)->GetLogicalVolume()->GetName()));
#ifdef DEBUG
      debugint.emplace_back(v->GetVolume(k)->GetCopyNo());
      debugstring.emplace_back(v->GetVolume(k)->GetLogicalVolume()->GetName());
#endif
    }
  }
#ifdef DEBUG
  LogDebug("TrackerSimDebugNumbering") << " G4 TrackerG4SimHitNumberingScheme " << debugint;
  for (u_int32_t jj = 0; jj < debugstring.size(); jj++)
    LogDebug("TrackerSimDebugNumbering") << " " << debugstring[jj];
#endif
}

unsigned int TrackerG4SimHitNumberingScheme::g4ToNumberingScheme(const G4VTouchable* v) {
  if (alreadySet_ == false) {
    buildAll();
  }
  TrackerG4SimHitNumberingScheme::Nav_Story st;
  touchToNavStory(v, st);

#ifdef DEBUG
  dumpG4VPV(v);
  LogDebug("TrackerSimDebugNumbering") << " Returning: " << directMap_[st];
#endif

  return directMap_[st];
}

void TrackerG4SimHitNumberingScheme::dumpG4VPV(const G4VTouchable* v) {
  int levels = v->GetHistoryDepth();

  LogDebug("TrackerSimDebugNumbering") << " NAME : " << v->GetVolume()->GetLogicalVolume()->GetName();
  for (int k = 0; k <= levels; k++) {
    LogDebug("TrackerSimInfoNumbering") << " Hist: " << v->GetVolume(k)->GetLogicalVolume()->GetName() << " Copy "
                                        << v->GetVolume(k)->GetCopyNo();
  }
}
