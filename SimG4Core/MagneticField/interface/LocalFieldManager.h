#ifndef SimG4Core_LocalFieldManager_H
#define SimG4Core_LocalFieldManager_H

#include "G4FieldManager.hh"

namespace sim {

  class LocalFieldManager : public G4FieldManager {
  public:
    // ctor & dtor
    LocalFieldManager() : G4FieldManager() {}
    ~LocalFieldManager() override {}
    LocalFieldManager(G4Field *commonField, G4FieldManager *priFM, G4FieldManager *altFM);
    void ConfigureForTrack(const G4Track *trk) override;
    void SetVerbosity(bool flag) {
      fVerbosity = flag;
      return;
    }

  protected:
    //
    const G4FieldManager *CopyValuesAndChordFinder(G4FieldManager *fm);
    void print(const G4Track *trk);

  private:
    G4FieldManager *fPrimaryFM;
    G4FieldManager *fAlternativeFM;
    G4FieldManager *fCurrentFM;
    bool fVerbosity;
  };

}  // namespace sim

#endif
