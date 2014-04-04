#ifndef Validation_Geometry_MaterialBudget_h
#define Validation_Geometry_MaterialBudget_h

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

class MaterialBudget : public SimWatcher, 
                       public Observer<const BeginOfRun*>,
                       public Observer<const BeginOfTrack*>,
		       public Observer<const G4Step*>,
                       public Observer<const EndOfTrack*> {

public:

  MaterialBudget(const edm::ParameterSet&);
  virtual ~MaterialBudget();
  
private:

  MaterialBudget(const MaterialBudget&);          // stop default
  const MaterialBudget& operator=(const MaterialBudget&); // ...
  
  void update(const BeginOfRun*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);

  void book(const edm::ParameterSet&);
  bool stopAfter(const G4Step*);
  
  std::vector<std::string>      detTypes, detNames;
  std::vector<int>              constituents, detLevels,regionTypes,stackOrder;
  std::vector<double>           etaRegions, boundaries;
  std::vector<G4LogicalVolume*> logVolumes;
  std::vector<TProfile*>        me100, me200, me300, me400, me500, me600;
  std::vector<double>           stepLen, radLen, intLen;
  double                        eta, phi, stepT;
};

#endif
