#ifndef Validation_Geometry_MaterialBudgetForward_h
#define Validation_Geometry_MaterialBudgetForward_h

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Vector/LorentzVector.h>

class BeginOfRun;
class BeginOfTrack;
class G4Step;
class EndOfTrack;
#include "G4LogicalVolume.hh"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <string>
#include <vector>

class MaterialBudgetForward : public SimWatcher,
                              public Observer<const BeginOfRun *>,
                              public Observer<const BeginOfTrack *>,
                              public Observer<const G4Step *>,
                              public Observer<const EndOfTrack *> {
public:
  MaterialBudgetForward(const edm::ParameterSet &);
  MaterialBudgetForward(const MaterialBudgetForward &) = delete;  // stop default
  ~MaterialBudgetForward() override;

  const MaterialBudgetForward &operator=(const MaterialBudgetForward &) = delete;  // ...

private:
  void update(const BeginOfRun *) override;
  void update(const BeginOfTrack *) override;
  void update(const G4Step *) override;
  void update(const EndOfTrack *) override;

  void book(const edm::ParameterSet &);
  bool stopAfter(const G4Step *);

  std::vector<std::string> detTypes, detNames;
  std::vector<int> constituents, detLevels, regionTypes, stackOrder;
  std::vector<double> etaRegions, boundaries;
  std::vector<G4LogicalVolume *> logVolumes;
  static const int maxSet = 25;
  TH1F *me400[maxSet];
  TH2F *me800[maxSet];
  TProfile *me100[maxSet], *me200[maxSet], *me300[maxSet];
  TProfile2D *me500[maxSet], *me600[maxSet], *me700[maxSet];
  std::vector<double> stepLen, radLen, intLen;
  double eta_, phi_, stepT;
};

#endif
