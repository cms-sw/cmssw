#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>

#include <string>

//
// class decleration
//

class BeginOfEvent;
class EndOfEvent;
class G4Step;


class GflashG4Watcher : public SimWatcher,
  public Observer<const BeginOfEvent*>,
  public Observer<const EndOfEvent*>,
  public Observer<const G4Step*> {

 public:
  GflashG4Watcher(const edm::ParameterSet& p);
  ~GflashG4Watcher();
  
 private:

  G4bool        inc_flag;
  G4double      inc_energy;
  G4ThreeVector inc_direction;
  G4ThreeVector inc_position;

  void update(const BeginOfEvent* );
  void update(const EndOfEvent* );
  void update(const G4Step* );

  // histograms for GflashG4Watcher

  std::string histFileName_;
  TFile*    histFile_;

  TH1F*     em_incE;
  TH1F*     em_ssp_rho;
  TH1F*     em_ssp_z;
  TH1F*     em_long;
  TH2F*     em_lateral;
  TH1F*     em_long_sd;
  TH2F*     em_lateral_sd;

};

