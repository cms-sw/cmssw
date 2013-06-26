#ifndef SimG4Core_CheckSecondary_CheckSecondary_H
#define SimG4Core_CheckSecondary_CheckSecondary_H

#include "SimG4Core/CheckSecondary/interface/TreatSecondary.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>

class G4Step;
class BeginOfEvent;
class BeginOfTrack;
class EndOfEvent;

class CheckSecondary : public SimWatcher,
		       public Observer<const BeginOfEvent *>, 
                       public Observer<const BeginOfTrack *>,
		       public Observer<const EndOfEvent *>, 
		       public Observer<const G4Step *> {

public:
  CheckSecondary(const edm::ParameterSet &p);
  virtual ~CheckSecondary();

private:
  CheckSecondary(const CheckSecondary&); // stop default
  const CheckSecondary& operator=(const CheckSecondary&);

  // observer classes
  TTree * bookTree(std::string);
  void endTree();
  void update(const BeginOfEvent * evt);
  void update(const BeginOfTrack * trk);
  void update(const G4Step * step);
  void update(const EndOfEvent * evt);

private:
  TreatSecondary                       *treatSecondary;
  G4ProcessTypeEnumerator              *typeEnumerator;
  bool                                 saveToTree, storeIt;
  int                                  nHad;
  std::vector<int>                     *nsec, *procids;
  std::vector<double>                  *px, *py, *pz, *mass, *deltae;
  std::vector<std::string>             *procs;
  TFile                                *file;
  TTree                                *tree;
};

#endif
