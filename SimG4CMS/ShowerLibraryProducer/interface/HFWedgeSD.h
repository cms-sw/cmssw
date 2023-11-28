#ifndef SimG4CMS_ShowerLibraryProducer_HFWedgeSD_h
#define SimG4CMS_ShowerLibraryProducer_HFWedgeSD_h

#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"

#include <map>

class G4Step;
class G4HCofThisEvent;

class HFWedgeSD : public SensitiveCaloDetector {
public:
  explicit HFWedgeSD(const std::string&, const SensitiveDetectorCatalog& clg, const SimTrackManager*);
  ~HFWedgeSD() override;

  void Initialize(G4HCofThisEvent* HCE) override;
  bool ProcessHits(G4Step* step, G4TouchableHistory* tHistory) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void clearHits() override;
  uint32_t setDetUnitId(const G4Step*) override;
  void fillHits(edm::PCaloHitContainer&, const std::string&) override;

protected:
  G4bool hitExists();
  HFShowerG4Hit* createNewHit();
  void updateHit(HFShowerG4Hit*);

private:
  int hcID;
  HFShowerG4HitsCollection* theHC;
  std::map<int, HFShowerG4Hit*> hitMap;

  int currentID, previousID, trackID;
  double edep, time;
  G4ThreeVector globalPos, localPos, momDir;
  HFShowerG4Hit* currentHit;
};

#endif
