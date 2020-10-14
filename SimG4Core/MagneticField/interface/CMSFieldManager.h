#ifndef SimG4Core_MagneticField_CMSFieldManager_H
#define SimG4Core_MagneticField_CMSFieldManager_H

/*
   Created:  13 January 2017, V. Ivanchenko
   This class implements smart magnetic field manager
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4FieldManager.hh"
#include <vector>

class G4Track;
class G4ChordFinder;
class G4PropagatorInField;
class G4MagIntegratorStepper;
class G4Region;

namespace sim {
  class Field;
}

class CMSFieldManager : public G4FieldManager {
public:
  explicit CMSFieldManager();

  ~CMSFieldManager() override;

  void ConfigureForTrack(const G4Track *) override;

  void InitialiseForVolume(const edm::ParameterSet &,
                           sim::Field *,
                           G4ChordFinder *cfDefault,
                           G4ChordFinder *cfMonopole,
                           const std::string &vol,
                           const std::string &fieldType,
                           const std::string &stepperName,
                           double delta,
                           G4PropagatorInField *);

  void setMonopoleTracking(G4bool);

  CMSFieldManager(const CMSFieldManager &) = delete;
  CMSFieldManager &operator=(const CMSFieldManager &) = delete;

private:
  bool isInsideVacuum(const G4Track *);
  bool isInsideTracker(const G4Track *);
  void setDefaultChordFinder();
  void setChordFinderForTracker();
  void setChordFinderForVacuum();

  std::unique_ptr<sim::Field> theField;

  G4ChordFinder *m_currChordFinder;
  G4ChordFinder *m_chordFinder;
  G4ChordFinder *m_chordFinderMonopole;

  G4PropagatorInField *m_propagator;

  std::vector<const G4Region *> m_regions;

  double m_dChord;
  double m_dChordTracker;
  double m_dOneStep;
  double m_dOneStepTracker;
  double m_dIntersection;
  double m_dInterTracker;
  double m_Rmax2;
  double m_Zmax;
  double m_stepMax;
  double m_energyThTracker;
  double m_energyThreshold;
  double m_dChordSimple;
  double m_dOneStepSimple;
  double m_dIntersectionSimple;
  double m_stepMaxSimple;

  bool m_cfTracker;
  bool m_cfVacuum;
};
#endif
