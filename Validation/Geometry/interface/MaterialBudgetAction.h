#ifndef Validation_Geometry_MaterialBudgetAction_h
#define Validation_Geometry_MaterialBudgetAction_h

#include <string>
#include <vector>

#include "Validation/Geometry/interface/MaterialBudgetTree.h"
#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/MaterialBudgetHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTrackerHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetEcalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTxt.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>

class BeginOfTrack;
class BeginOfRun;
class BeginOfEvent;
class G4Step;
class EndOfTrack;
class G4StepPoint;
class G4VTouchable;

class MaterialBudgetAction : public SimWatcher, 
                             public Observer<const BeginOfRun*>,
			     public Observer<const BeginOfTrack*>,
			     public Observer<const G4Step*>,
			     public Observer<const EndOfTrack*>
{
 public:
  MaterialBudgetAction(const edm::ParameterSet&);
  virtual ~MaterialBudgetAction();
  
 private:
  MaterialBudgetAction(const MaterialBudgetAction&); // stop default
  
  const MaterialBudgetAction& operator=(const MaterialBudgetAction&); // stop default
  
  void update(const BeginOfRun*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);
  
  bool CheckTouchableInSelectedVolumes( const G4VTouchable* touch );
  bool StopAfterProcess( const G4Step* aStep );

  void save( const G4Step* aStep );
  std::string getSubDetectorName( G4StepPoint* aStepPoint );
  std::string getPartName( G4StepPoint* aStepPoint );
  MaterialBudgetData* theData;
  MaterialBudgetTree* theTree;
  MaterialBudgetFormat* theHistos;
  MaterialBudgetTxt* theTxt;
  TestHistoMgr* theHistoMgr;
  bool saveToTxt, saveToTree, saveToHistos;
  bool storeDecay;
  double Ekin;
  bool firstParticle;
  
  std::vector<G4String> theVolumeList; 
  G4String theProcessToStop;
  std::string theHistoList;
};

#endif
