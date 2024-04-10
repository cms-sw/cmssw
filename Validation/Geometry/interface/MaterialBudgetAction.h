#ifndef Validation_Geometry_MaterialBudgetAction_h
#define Validation_Geometry_MaterialBudgetAction_h

#include <string>
#include <vector>

#include "Validation/Geometry/interface/MaterialBudgetTree.h"
#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/MaterialBudgetHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTrackerHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetEcalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetMtdHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetHGCalHistos.h"
#include "Validation/Geometry/interface/MaterialBudgetTxt.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>
#include <G4VTouchable.hh>

class BeginOfTrack;
class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;
class G4Step;
class EndOfTrack;
class EndOfRun;
class G4StepPoint;

class MaterialBudgetAction : public SimWatcher,
                             public Observer<const BeginOfRun*>,
                             public Observer<const BeginOfTrack*>,
                             public Observer<const G4Step*>,
                             public Observer<const EndOfTrack*>,
                             public Observer<const EndOfRun*> {
public:
  MaterialBudgetAction(const edm::ParameterSet&);
  ~MaterialBudgetAction() override;

private:
  MaterialBudgetAction(const MaterialBudgetAction&);  // stop default

  const MaterialBudgetAction& operator=(const MaterialBudgetAction&);  // stop default

  void update(const BeginOfRun*) override;
  void update(const BeginOfTrack*) override;
  void update(const G4Step*) override;
  void update(const EndOfTrack*) override;
  void update(const EndOfRun*) override;

  bool CheckTouchableInSelectedVolumes(const G4VTouchable* touch);
  bool StopAfterProcess(const G4Step* aStep);

  void save(const G4Step* aStep);
  std::string getSubDetectorName(G4StepPoint* aStepPoint);
  std::string getPartName(G4StepPoint* aStepPoint);

  std::shared_ptr<MaterialBudgetData> theData;
  std::shared_ptr<MaterialBudgetTree> theTree;
  std::shared_ptr<MaterialBudgetFormat> theHistos;
  std::shared_ptr<MaterialBudgetTxt> theTxt;
  std::shared_ptr<TestHistoMgr> theHistoMgr;

  bool saveToTxt, saveToTree, saveToHistos;
  bool storeDecay;
  double Ekin;
  bool firstParticle;

  std::vector<G4String> theVolumeList;
  G4String theProcessToStop;
  std::string theHistoList;
};

#endif
