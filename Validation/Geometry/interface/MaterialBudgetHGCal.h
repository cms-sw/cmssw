#ifndef Validation_Geometry_MaterialBudgetHGCal_h
#define Validation_Geometry_MaterialBudgetHGCal_h

#include "Validation/Geometry/interface/MaterialBudgetHGCalHistos.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>

class BeginOfJob;
class BeginOfTrack;
class G4Step;
class EndOfTrack;

class MaterialBudgetHGCal
: public SimWatcher, 
  public Observer<const BeginOfJob*>,
  public Observer<const BeginOfTrack*>,
  public Observer<const G4Step*>,
  public Observer<const EndOfTrack*>
{
 public:

  MaterialBudgetHGCal( const edm::ParameterSet& );
  virtual ~MaterialBudgetHGCal( void );
  
 private:

  MaterialBudgetHGCal( const MaterialBudgetHGCal& );
  const MaterialBudgetHGCal& operator=( const MaterialBudgetHGCal& );
  
  void update( const BeginOfJob* );
  void update( const BeginOfTrack* );
  void update( const G4Step* );
  void update( const EndOfTrack* );

  bool stopAfter( const G4Step* );
  
  MaterialBudgetHGCalHistos* m_HistoHGCal;
  double m_rMax;
  double m_zMax;
};

#endif
