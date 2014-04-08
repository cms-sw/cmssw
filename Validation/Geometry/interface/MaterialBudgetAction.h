#ifndef _MaterialBudgetAction_h
#define _MaterialBudgetAction_h
#include <string>
#include <vector>
#include <map>
 
// user include files
#include "Validation/Geometry/interface/MaterialBudgetTree.h"
#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/MaterialBudgetHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTrackerHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetEcalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTxt.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>

class BeginOfTrack;
class BeginOfRun;
class G4Step;
class EndOfTrack;
class EndOfEvent;
class G4StepPoint;
class G4VTouchable;

class MaterialBudgetAction : public SimProducer, 
			     public Observer<const BeginOfRun*>,
			     public Observer<const BeginOfTrack*>,
			     public Observer<const G4Step*>,
			     public Observer<const EndOfTrack*>,
			     public Observer<const EndOfEvent *>
{
 public:
  MaterialBudgetAction(const edm::ParameterSet&);
  virtual ~MaterialBudgetAction();
  
  void produce(edm::Event&, const edm::EventSetup&);
  
  
 private:
  MaterialBudgetAction(const MaterialBudgetAction&); // stop default
  
  const MaterialBudgetAction& operator=(const MaterialBudgetAction&); // stop default
  
  void update(const BeginOfRun*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);
  void update(const EndOfEvent*);
  
  void initRun();
  void processEvent( unsigned int nEv );
  void endRun();
  
  bool CheckTouchableInSelectedVolumes( const G4VTouchable* touch );
  bool StopAfterProcess( const G4Step* aStep );

 private:
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
