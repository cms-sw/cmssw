#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4LogicalVolume.hh"
#include "G4Region.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"

#include <string>
#include <vector>

class EventAction;
class G4VTouchable;
class G4Track;

class SteppingAction: public G4UserSteppingAction {

public:
  SteppingAction(EventAction * ea,const edm::ParameterSet & ps);
  virtual ~SteppingAction();

  virtual void UserSteppingAction(const G4Step * aStep);
  
  SimActivityRegistry::G4StepSignal m_g4StepSignal;

private:

  bool initPointer();

  bool catchLowEnergyInVacuum(G4Track * theTrack) const; 
  bool catchLongLived(const G4Step * aStep) const;
  bool killLowEnergy(const G4Step * aStep) const;
  bool isThisVolume(const G4VTouchable* touch, G4VPhysicalVolume* pv) const;

  void PrintKilledTrack(const G4Track*, int type) const;

private:

  EventAction                   *eventAction_;
  G4VPhysicalVolume             *tracker, *calo;
  double                        theCriticalEnergyForVacuum;
  double                        theCriticalDensity;
  double                        maxTrackTime;
  std::vector<double>           maxTrackTimes, ekinMins;
  std::vector<std::string>      maxTimeNames, ekinNames, ekinParticles;
  std::vector<const G4Region*>  maxTimeRegions;
  std::vector<G4LogicalVolume*> ekinVolumes;
  std::vector<int>              ekinPDG;
  unsigned int                  numberTimes;
  unsigned int                  numberEkins;
  unsigned int                  numberPart;

  bool                          initialized;
  bool                          killBeamPipe;

};

#endif
