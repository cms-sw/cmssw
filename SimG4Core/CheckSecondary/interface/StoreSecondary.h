#ifndef SimG4Core_CheckSecondary_StoreSecondary_H
#define SimG4Core_CheckSecondary_StoreSecondary_H

#include "SimG4Core/CheckSecondary/interface/TreatSecondary.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>

class G4Step;
class BeginOfEvent;
class BeginOfTrack;

class StoreSecondary : public SimProducer,
		       public Observer<const BeginOfEvent *>, 
                       public Observer<const BeginOfTrack *>,
		       public Observer<const G4Step *> {

public:
  StoreSecondary(const edm::ParameterSet &p);
  virtual ~StoreSecondary();

  void produce(edm::Event&, const edm::EventSetup&);

private:
  StoreSecondary(const StoreSecondary&); // stop default
  const StoreSecondary& operator=(const StoreSecondary&);

  // observer classes
  void update(const BeginOfEvent * evt);
  void update(const BeginOfTrack * trk);
  void update(const G4Step * step);

private:
  int                  verbosity, killAfter;
  int                  nHad;
  bool                 storeIt;
  std::vector<math::XYZTLorentzVector> secondaries;
  std::vector<int>                     nsecs;
  std::vector<std::string>             procs;
  TreatSecondary*                      treatSecondary;
};

#endif
