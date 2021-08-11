#ifndef SimG4Core_CheckSecondary_TreatSecondary_H
#define SimG4Core_CheckSecondary_TreatSecondary_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

class G4Step;
class G4Track;
class G4ProcessTypeEnumerator;

class TreatSecondary {
public:
  TreatSecondary(const edm::ParameterSet &p);
  TreatSecondary(const TreatSecondary &) = delete;  // stop default
  const TreatSecondary &operator=(const TreatSecondary &) = delete;
  virtual ~TreatSecondary();

  void initTrack(const G4Track *trk);
  std::vector<math::XYZTLorentzVector> tracks(
      const G4Step *step, std::string &procName, int &procID, bool &intr, double &deltaE, std::vector<int> &charges);

private:
  int verbosity, minSec, killAfter;
  double minDeltaE, eTrack;
  G4ProcessTypeEnumerator *typeEnumerator;
  int step, nHad, nsecL;
};

#endif
