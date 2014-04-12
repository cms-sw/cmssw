#ifndef SimG4CMS_ShowerLibraryProducer_HFChamberSD_h
#define SimG4CMS_ShowerLibraryProducer_HFChamberSD_h

#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

#include <iostream>
#include <fstream>
#include <vector>

class G4Step;
class G4HCofThisEvent;

class HFChamberSD : public SensitiveCaloDetector {

public:

  HFChamberSD(std::string, const DDCompactView&, SensitiveDetectorCatalog&,
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~HFChamberSD();

  virtual void     Initialize(G4HCofThisEvent*HCE);
  virtual G4bool   ProcessHits(G4Step* aStep,G4TouchableHistory* ROhist);
  virtual void     EndOfEvent(G4HCofThisEvent* HCE);
  virtual void     clear();
  virtual void     DrawAll();
  virtual void     PrintAll();

protected:

  virtual void     clearHits();
  virtual uint32_t setDetUnitId(G4Step*);
  virtual void     fillHits(edm::PCaloHitContainer&, std::string);

private:

  std::string               theName;
  const SimTrackManager*    m_trackManager;

  G4int                     theHCID;
  HFShowerG4HitsCollection* theHC;
  int                       theNSteps;
};

#endif

