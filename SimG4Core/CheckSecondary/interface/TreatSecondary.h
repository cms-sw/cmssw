#ifndef SimG4Core_CheckSecondary_TreatSecondary_H
#define SimG4Core_CheckSecondary_TreatSecondary_H

#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>

class G4Step;
class G4Track;

class TreatSecondary {

public:
  TreatSecondary(const edm::ParameterSet &p);
  virtual ~TreatSecondary();

  void initTrack(const G4Track* trk);
  std::vector<math::XYZTLorentzVector> tracks(const G4Step * step, 
					      std::string & procName, 
					      int & procID, bool & intr,
					      double & deltaE,
					      std::vector<int> & charges);

private:
  TreatSecondary(const TreatSecondary&); // stop default
  const TreatSecondary& operator=(const TreatSecondary&);

private:
  int                                  verbosity, minSec, killAfter;
  double                               minDeltaE, eTrack;
  G4ProcessTypeEnumerator              *typeEnumerator;
  int                                  step, nHad, nsecL;
};

#endif
