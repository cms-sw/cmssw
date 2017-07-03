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
	    const SensitiveDetectorCatalog & clg,
	    edm::ParameterSet const & p, const SimTrackManager*);
  ~HFWedgeSD() override;
  
  void     Initialize(G4HCofThisEvent * HCE) override;
  bool     ProcessHits(G4Step * step,G4TouchableHistory * tHistory) override;
  void     EndOfEvent(G4HCofThisEvent * eventHC) override;
  void     clear() override;
  void     DrawAll() override;
  void     PrintAll() override;

protected:

  G4bool           hitExists();
  HFShowerG4Hit*   createNewHit();
  void             updateHit(HFShowerG4Hit*);

  void     clearHits() override;
  uint32_t setDetUnitId(G4Step*) override;
  void     fillHits(edm::PCaloHitContainer&, std::string) override;


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
