#ifndef SimG4Core_ChordFinderSetter_H
#define SimG4Core_ChordFinderSetter_H

class G4ChordFinder;
class G4FieldManager;

namespace sim {
  class ChordFinderSetter {
  public:
    ChordFinderSetter();
    ~ChordFinderSetter();

    bool isMonopoleSet() const { return fChordFinderMonopole; }
    void setMonopole(G4ChordFinder *cfm) { fChordFinderMonopole = cfm; }

    void setStepperAndChordFinder(G4FieldManager * fM, int val);

  private:
    static thread_local G4ChordFinder *fChordFinder, *fChordFinderMonopole;
  };
};

#endif
