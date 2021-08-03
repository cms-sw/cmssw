#ifndef SimG4Core_CheckSecondary_StoreSecondary_H
#define SimG4Core_CheckSecondary_StoreSecondary_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class G4Step;
class BeginOfEvent;
class BeginOfTrack;
class TreatSecondary;

class StoreSecondary : public SimProducer,
                       public Observer<const BeginOfEvent *>,
                       public Observer<const BeginOfTrack *>,
                       public Observer<const G4Step *> {
public:
  StoreSecondary(const edm::ParameterSet &p);
  StoreSecondary(const StoreSecondary &) = delete;  // stop default
  const StoreSecondary &operator=(const StoreSecondary &) = delete;
  ~StoreSecondary() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  // observer classes
  void update(const BeginOfEvent *evt) override;
  void update(const BeginOfTrack *trk) override;
  void update(const G4Step *step) override;

private:
  int verbosity, killAfter;
  int nHad;
  bool storeIt;
  std::vector<math::XYZTLorentzVector> secondaries;
  std::vector<int> nsecs;
  std::vector<std::string> procs;
  TreatSecondary *treatSecondary;
};

#endif
