#ifndef SimG4CMS_ShowerLibraryProducer_FiberSensitiveDetector_h
#define SimG4CMS_ShowerLibraryProducer_FiberSensitiveDetector_h

#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/FiberG4Hit.h"
#include "SimG4CMS/Calo/interface/HFShower.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

#include <iostream>
#include <fstream>
#include <vector>

class G4Step;
class G4HCofThisEvent;

class FiberSensitiveDetector : public SensitiveCaloDetector,
                               public Observer<const BeginOfJob *>,
                               public Observer<const BeginOfRun *>,
                               public Observer<const BeginOfEvent *>,
                               public Observer<const EndOfEvent *> {
public:
  explicit FiberSensitiveDetector(const std::string &,
                                  const HcalSimulationConstants *,
                                  const HcalDDDSimConstants *,
                                  const SensitiveDetectorCatalog &,
                                  edm::ParameterSet const &,
                                  const SimTrackManager *);
  ~FiberSensitiveDetector() override;

  void Initialize(G4HCofThisEvent *HCE) override;
  G4bool ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist) override;
  void EndOfEvent(G4HCofThisEvent *HCE) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void clearHits() override;
  uint32_t setDetUnitId(const G4Step *) override;
  void fillHits(edm::PCaloHitContainer &, const std::string &) override;

protected:
  void update(const BeginOfJob *) override;
  void update(const BeginOfRun *) override;
  void update(const BeginOfEvent *) override;
  void update(const ::EndOfEvent *) override;

private:
  HFShower *theShower;

  G4int theHCID;
  FiberG4HitsCollection *theHC;
};

#endif
