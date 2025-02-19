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
  virtual ~MaterialBudgetHcal();
  
private:

  MaterialBudgetHcal(const MaterialBudgetHcal&); // stop default
  const MaterialBudgetHcal& operator=(const MaterialBudgetHcal&); // stop default
  
  void update(const BeginOfJob*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);

  bool stopAfter(const G4Step*);
  
  MaterialBudgetHcalHistos*   theHistoHcal;
  MaterialBudgetCastorHistos* theHistoCastor;
  double                      rMax, zMax;
};

#endif
