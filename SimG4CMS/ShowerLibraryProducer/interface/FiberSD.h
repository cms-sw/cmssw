#ifndef SimG4CMS_ShowerLibraryProducer_FiberSD_h
#define SimG4CMS_ShowerLibraryProducer_FiberSD_h

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/FiberG4Hit.h"
#include "SimG4CMS/Calo/interface/HFShower.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

#include <iostream>
#include <fstream>
#include <vector>

class G4Step;
class G4HCofThisEvent;

class FiberSD : public SensitiveCaloDetector,
                public Observer<const BeginOfJob *>,
                public Observer<const BeginOfRun *>,
		public Observer<const BeginOfEvent*>,
		public Observer<const EndOfEvent*> {

public:

  FiberSD(std::string, const DDCompactView&, const SensitiveDetectorCatalog&,
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~FiberSD();

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

  virtual void     update(const BeginOfJob *);
  virtual void     update(const BeginOfRun *);
  virtual void     update(const BeginOfEvent *);
  virtual void     update(const ::EndOfEvent *);

private:

  std::string            theName;
  const SimTrackManager* m_trackManager;
  HFShower*              theShower;

  G4int                  theHCID;
  FiberG4HitsCollection* theHC;
};

#endif

