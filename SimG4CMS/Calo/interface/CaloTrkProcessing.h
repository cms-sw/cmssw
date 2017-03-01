#ifndef SimG4CMS_CaloTrkProcessing_H
#define SimG4CMS_CaloTrkProcessing_H

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VTouchable.hh"

#include <map>
#include <vector>
#include <string>
#include <iostream>

class SimTrackManager;
class BeginOfEvent;
class G4LogicalVolume;
class G4Step;

class CaloTrkProcessing : public SensitiveCaloDetector, 
			  public Observer<const BeginOfEvent *>,
			  public Observer<const G4Step *> {

public:

  CaloTrkProcessing(G4String aSDname, const DDCompactView & cpv,
		    const SensitiveDetectorCatalog & clg,
		    edm::ParameterSet const & p, const SimTrackManager*);
  virtual ~CaloTrkProcessing();
  virtual void             Initialize(G4HCofThisEvent * ) {}
  virtual void             clearHits() {}
  virtual bool             ProcessHits(G4Step * , G4TouchableHistory * ) {
    return true;
  }
  virtual uint32_t         setDetUnitId(G4Step * step) {return 0;}
  virtual void             EndOfEvent(G4HCofThisEvent * ) {}
  void                     fillHits(edm::PCaloHitContainer&, std::string ) {}

private:

  void                     update(const BeginOfEvent * evt);
  void                     update(const G4Step *);
  std::vector<std::string> getNames(G4String, const DDsvalues_type&);
  std::vector<double>      getNumbers(G4String, const DDsvalues_type&);
  int                      isItCalo(const G4VTouchable*);
  int                      isItInside(const G4VTouchable*, int, int);


  struct Detector {
    Detector() {}
    std::string                   name;
    G4LogicalVolume*              lv;
    int                           level;
    std::vector<std::string>      fromDets; 
    std::vector<G4LogicalVolume*> fromDetL;
    std::vector<int>              fromLevels;
  };
  
  // Utilities to get detector levels during a step
  int                      detLevels(const G4VTouchable*) const;
  G4LogicalVolume*         detLV(const G4VTouchable*, int) const;
  void                     detectorLevel(const G4VTouchable*, int&, int*,
					 G4String*) const;

  bool                     testBeam, putHistory;
  double                   eMin;
  int                      lastTrackID;
  std::vector<Detector>    detectors;
  const SimTrackManager*   m_trackManager;
};

#endif








