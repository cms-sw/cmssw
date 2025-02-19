#ifndef SimG4Core_FieldStepWatcher_H
#define SimG4Core_FieldStepWatcher_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "G4NavigationHistory.hh"

#include <iostream>
#include <vector>
#include <map>
#include <string>

class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;
class G4Step;

class FieldStepWatcher : public SimWatcher,
                         public Observer<const BeginOfRun *>,
                         public Observer<const BeginOfEvent *>,
                         public Observer<const EndOfEvent *>,
                         public Observer<const G4Step *> {

public:
  FieldStepWatcher(edm::ParameterSet const & p);
  ~FieldStepWatcher();
private:
  void update(const BeginOfRun * );
  void update(const BeginOfEvent * );
  void update(const EndOfEvent * );
  void update(const G4Step * );
  void findTouch(G4VPhysicalVolume * , int );
  int  findName(std::string);
private:
  int                          level;
  std::string                  outFile;

  std::vector<std::string>     lvnames;
  std::vector<int>             steps;
  G4NavigationHistory          fHistory;

  DQMStore                     *dbe_;
  std::vector<MonitorElement*> meStep, meCall, meStepCH, meStepNH, meStepC;
  std::vector<MonitorElement*> meStepE, meStepG, meStepMu, meStepNu, meStepN;
};

#endif
