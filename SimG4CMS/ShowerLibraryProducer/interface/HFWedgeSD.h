#ifndef SimG4CMS_ShowerLibraryProducer_HFWedgeSD_h
#define SimG4CMS_ShowerLibraryProducer_HFWedgeSD_h

#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

 
#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class G4Step;
class G4HCofThisEvent;

class HFWedgeSD : public SensitiveCaloDetector {

public:    
  
  HFWedgeSD(std::string name, const DDCompactView & cpv,
	    SensitiveDetectorCatalog & clg, 
	    edm::ParameterSet const & p, const SimTrackManager*);
  virtual ~HFWedgeSD();
  
  virtual void     Initialize(G4HCofThisEvent * HCE);
  virtual bool     ProcessHits(G4Step * step,G4TouchableHistory * tHistory);
  virtual void     EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void     clear();
  virtual void     DrawAll();
  virtual void     PrintAll();

protected:

  G4bool           hitExists();
  HFShowerG4Hit*   createNewHit();
  void             updateHit(HFShowerG4Hit*);

  virtual void     clearHits();
  virtual uint32_t setDetUnitId(G4Step*);
  virtual void     fillHits(edm::PCaloHitContainer&, std::string);


private:

  std::string                  theName;
  const SimTrackManager*       m_trackManager;

  int                          hcID;
  HFShowerG4HitsCollection*    theHC; 
  std::map<int,HFShowerG4Hit*> hitMap;

  int                          currentID, previousID, trackID;
  double                       edep, time;
  G4ThreeVector                globalPos, localPos, momDir;
  HFShowerG4Hit*               currentHit;
};

#endif 
