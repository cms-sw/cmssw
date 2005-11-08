///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.cc
// Description: Base class for numbering scheme of calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"
#include "globals.hh"

CaloNumberingScheme::CaloNumberingScheme(int iv) : verbosity(iv) {}

void CaloNumberingScheme::setVerbosity(const int iv) {verbosity = iv;}

int CaloNumberingScheme::detectorLevel(const G4Step* aStep) const {

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  return level;
}

void CaloNumberingScheme::detectorLevel(const G4Step* aStep, int& level,
					int* copyno, G4String* name) const {

  //Get name and copy numbers
  if (level > 0) {
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      name[ii]   = touch->GetVolume(i)->GetName();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
