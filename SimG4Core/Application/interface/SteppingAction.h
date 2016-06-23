#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4LogicalVolume.hh"
#include "G4Region.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"

#include <string>
#include <vector>

class EventAction;
class G4VTouchable;
class CMSSteppingVerbose;

class SteppingAction: public G4UserSteppingAction {

public:
  SteppingAction(EventAction * ea,const edm::ParameterSet & ps, const CMSSteppingVerbose*);
  virtual ~SteppingAction();

  virtual void UserSteppingAction(const G4Step * aStep);
  
  SimActivityRegistry::G4StepSignal m_g4StepSignal;

private:

  bool initPointer();

  bool killInsideDeadRegion(G4Track * theTrack, const G4Region* reg) const;
  bool catchLongLived(G4Track* theTrack, const G4Region* reg) const;
  bool killLowEnergy(const G4Step * aStep) const;

  bool isThisVolume(const G4VTouchable* touch, G4VPhysicalVolume* pv) const;
  void PrintKilledTrack(const G4Track*, const std::string&) const;

private:

  EventAction                   *eventAction_;
  G4VPhysicalVolume             *tracker, *calo;
  const CMSSteppingVerbose*     steppingVerbose;
  double                        theCriticalEnergyForVacuum;
  double                        theCriticalDensity;
  double                        maxTrackTime;
  std::vector<double>           maxTrackTimes, ekinMins;
  std::vector<std::string>      maxTimeNames, ekinNames, ekinParticles;
  std::vector<std::string>      deadRegionNames;
  std::vector<const G4Region*>  maxTimeRegions;
  std::vector<const G4Region*>  deadRegions;
  std::vector<G4LogicalVolume*> ekinVolumes;
  std::vector<int>              ekinPDG;
  unsigned int                  numberTimes;
  unsigned int                  numberEkins;
  unsigned int                  numberPart;
  unsigned int                  ndeadRegions;

  bool                          initialized;
  bool                          killBeamPipe;

};

#endif
