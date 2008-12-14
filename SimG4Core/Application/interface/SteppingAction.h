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

class SteppingAction: public G4UserSteppingAction {

public:
  SteppingAction(EventAction * ea,const edm::ParameterSet & ps);
  ~SteppingAction();
  void UserSteppingAction(const G4Step * aStep);
  
  SimActivityRegistry::G4StepSignal m_g4StepSignal;
private:
  void catchLowEnergyInVacuumHere(const G4Step * aStep);
  void catchLowEnergyInVacuumNext(const G4Step * aStep);
  bool catchLongLived            (const G4Step * aStep);
  bool killLowEnergy             (const G4Step * aStep);
  bool initPointer();
  bool isThisVolume(const G4VTouchable* touch, G4VPhysicalVolume* pv);
  void killTrack                 (const G4Step * aStep);
private:
  EventAction                   *eventAction_;
  bool                          initialized;
  G4VPhysicalVolume             *tracker, *calo;
  bool                          killBeamPipe;
  double                        theCriticalEnergyForVacuum;
  double                        theCriticalDensity;
  double                        maxTrackTime;
  std::vector<double>           maxTrackTimes, ekinMins;
  std::vector<std::string>      maxTimeNames, ekinNames, ekinParticles;
  std::vector<G4Region*>        maxTimeRegions;
  std::vector<G4LogicalVolume*> ekinVolumes;
  std::vector<int>              ekinPDG;
  int                           verbose;
};

#endif
