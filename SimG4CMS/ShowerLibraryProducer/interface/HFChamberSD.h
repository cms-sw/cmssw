#ifndef SimG4CMS_ShowerLibraryProducer_HFChamberSD_h
#define SimG4CMS_ShowerLibraryProducer_HFChamberSD_h

#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

#include <iostream>
#include <fstream>
#include <vector>

class G4Step;
class G4HCofThisEvent;

class HFChamberSD : public SensitiveCaloDetector {
public:
  explicit HFChamberSD(const std::string&, const SensitiveDetectorCatalog&, const SimTrackManager*);
  ~HFChamberSD() override;

  void Initialize(G4HCofThisEvent* HCE) override;
  G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;
  void EndOfEvent(G4HCofThisEvent* HCE) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void clearHits() override;
  uint32_t setDetUnitId(const G4Step*) override;
  void fillHits(edm::PCaloHitContainer&, const std::string&) override;

private:
  G4int theHCID;
  HFShowerG4HitsCollection* theHC;
  int theNSteps;
};

#endif
