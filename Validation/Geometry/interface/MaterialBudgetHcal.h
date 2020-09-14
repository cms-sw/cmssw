#ifndef Validation_Geometry_MaterialBudgetHcal_h
#define Validation_Geometry_MaterialBudgetHcal_h

#include "Validation/Geometry/interface/MaterialBudgetHcalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetCastorHistos.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>

class BeginOfJob;
class BeginOfTrack;
class G4Step;
class EndOfTrack;

class MaterialBudgetHcal : public SimWatcher,
                           public Observer<const BeginOfJob*>,
                           public Observer<const BeginOfTrack*>,
                           public Observer<const G4Step*>,
                           public Observer<const EndOfTrack*> {
public:
  MaterialBudgetHcal(const edm::ParameterSet&);
  MaterialBudgetHcal(const MaterialBudgetHcal&) = delete;  // stop default

  const MaterialBudgetHcal& operator=(const MaterialBudgetHcal&) = delete;  // stop default

private:
  void update(const BeginOfJob*) override;
  void update(const BeginOfTrack*) override;
  void update(const G4Step*) override;
  void update(const EndOfTrack*) override;

  bool stopAfter(const G4Step*);

  std::unique_ptr<MaterialBudgetHcalHistos> theHistoHcal_;
  std::unique_ptr<MaterialBudgetCastorHistos> theHistoCastor_;
  double rMax_, zMax_;
  bool fromdd4hep_;
};

#endif
