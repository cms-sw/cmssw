#ifndef SimG4Core_PrintSensitive_H
#define SimG4Core_PrintSensitive_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4NavigationHistory.hh"

#include <iostream>
#include <string>

class BeginOfRun;
class G4VPhysicalVolume;

class PrintSensitive : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  PrintSensitive(edm::ParameterSet const &p);
  ~PrintSensitive() override;

private:
  void update(const BeginOfRun *run) override;
  int dumpTouch(G4VPhysicalVolume *pv, unsigned int leafDepth, bool printIt, int ns, std::ostream &out = std::cout);
  G4VPhysicalVolume *getTopPV();

private:
  std::string name_;
  int nchar_;
  G4NavigationHistory fHistory;
};

#endif
