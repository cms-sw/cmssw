#include "SimG4Core/MagneticField/interface/ChordFinderSetter.h"

#include "G4FieldManager.hh"

namespace sim {
  thread_local G4ChordFinder *ChordFinderSetter::fChordFinder = nullptr;
  thread_local G4ChordFinder *ChordFinderSetter::fChordFinderMonopole = nullptr;

  ChordFinderSetter::ChordFinderSetter() {}
  ChordFinderSetter::~ChordFinderSetter() {}

  void ChordFinderSetter::setStepperAndChordFinder(G4FieldManager * fM, int val) {
    if (fM != 0) {
      if (val == 0) {
        if (fChordFinder != 0) fM->SetChordFinder(fChordFinder);
      } else {
        fChordFinder = fM->GetChordFinder();
        if (fChordFinderMonopole != 0) fM->SetChordFinder(fChordFinderMonopole);
      }
    }
  }
}
