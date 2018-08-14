#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include "globals.hh"
#include "G4ThreeVector.hh"
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
  ~GflashG4Watcher() override;
  
 private:

  G4bool        inc_flag;
  G4double      inc_energy;
  G4double      out_energy;
  G4ThreeVector inc_vertex;
  G4ThreeVector inc_direction;
  G4ThreeVector inc_position;

  void update(const BeginOfEvent* ) override;
  void update(const EndOfEvent* ) override;
  void update(const G4Step* ) override;

  // histograms for GflashG4Watcher

  std::string histFileName_;
  double recoEnergyScaleEB_;
  double recoEnergyScaleEE_;

  TFile*    histFile_;

  TH1F*     em_incE;
  TH1F*     em_vtx_rho;
  TH1F*     em_vtx_z;

  TH1F*     eb_ssp_rho;
  TH1F*     eb_hit_long;
  TH1F*     eb_hit_lat;
  TH2F*     eb_hit_rz;
  TH1F*     eb_hit_long_sd;
  TH1F*     eb_hit_lat_sd;
  TH2F*     eb_hit_rz_sd;

  TH1F*     ee_ssp_z;
  TH1F*     ee_hit_long;
  TH1F*     ee_hit_lat;
  TH2F*     ee_hit_rz;
  TH1F*     ee_hit_long_sd;
  TH1F*     ee_hit_lat_sd;
  TH2F*     ee_hit_rz_sd;

};

