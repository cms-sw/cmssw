#ifndef SimG4Core_CheckSecondary_CheckSecondary_H
#define SimG4Core_CheckSecondary_CheckSecondary_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class G4Step;
class BeginOfEvent;
class BeginOfTrack;
class EndOfEvent;
class TreatSecondary;
class ProcessTypeEnumerator;

class CheckSecondary : public SimWatcher,
                       public Observer<const BeginOfEvent *>,
                       public Observer<const BeginOfTrack *>,
                       public Observer<const EndOfEvent *>,
                       public Observer<const G4Step *> {
public:
  CheckSecondary(const edm::ParameterSet &p);
  CheckSecondary(const CheckSecondary &) = delete;  // stop default
  const CheckSecondary &operator=(const CheckSecondary &) = delete;
  ~CheckSecondary() override;

private:
  // observer classes
  TTree *bookTree(std::string);
  void endTree();
  void update(const BeginOfEvent *evt) override;
  void update(const BeginOfTrack *trk) override;
  void update(const G4Step *step) override;
  void update(const EndOfEvent *evt) override;

private:
  TreatSecondary *treatSecondary;
  G4ProcessTypeEnumerator *typeEnumerator;
  bool saveToTree, storeIt;
  int nHad;
  std::vector<int> *nsec, *procids;
  std::vector<double> *px, *py, *pz, *mass, *deltae;
  std::vector<std::string> *procs;
  TFile *file;
  TTree *tree;
};

#endif
