#ifndef SimG4Core_HelpfulWatchers_MonopoleSteppingAction_H
#define SimG4Core_HelpfulWatchers_MonopoleSteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "G4Step.hh"
#include <vector>

class BeginOfRun;
class BeginOfTrack;

class MonopoleSteppingAction : public SimWatcher,
                               public Observer<const BeginOfRun *>,
                               public Observer<const BeginOfTrack *>,
                               public Observer<const G4Step *> {
public:
  MonopoleSteppingAction(edm::ParameterSet const &p);
  ~MonopoleSteppingAction() override;
  void registerConsumes(edm::ConsumesCollector) override;
  void beginRun(edm::EventSetup const &) override;
  void update(const BeginOfRun *) override;
  void update(const BeginOfTrack *) override;
  void update(const G4Step *) override;

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_bFieldH_;
  bool mode, actOnTrack;
  std::vector<int> pdgCode;
  double eStart, pxStart, pyStart, pzStart;
  double dirxStart, diryStart, dirzStart;
  double cMevToJ, cMeVToKgMByS, cInMByS, magCharge, bZ;
};

#endif
