#ifndef SimG4Core_MagneticField_CMSFieldManager_H
#define SimG4Core_MagneticField_CMSFieldManager_H

/*
   Created:  13 January 2017, V. Ivanchenko 
   This class implements smart magnetic field manager 
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4FieldManager.hh"

class G4Track;
class G4ChordFinder;
namespace sim { class Field; }

class CMSFieldManager : public G4FieldManager
{
public:

  explicit CMSFieldManager();

  ~CMSFieldManager() override;

  void ConfigureForTrack(const G4Track*) override; 

  void InitialiseForVolume(const edm::ParameterSet&, sim::Field*, G4ChordFinder*, G4ChordFinder*, 
                           const std::string& vol, const std::string& type, 
                           const std::string& stepper, double delta, double minstep);

  void SetMonopoleTracking(G4bool);

private:

  CMSFieldManager(const CMSFieldManager&) = delete;
  CMSFieldManager& operator=(const CMSFieldManager&) = delete;

  std::unique_ptr<sim::Field> theField;

  G4ChordFinder* currChordFinder;
  G4ChordFinder* chordFinder;
  G4ChordFinder* chordFinderMonopole;

  double dChord;
  double dOneStep;
  double dIntersection;
  double energyThreshold;
  double dChordSimple;
  double dOneStepSimple;
  double dIntersectionSimple;
};
#endif
