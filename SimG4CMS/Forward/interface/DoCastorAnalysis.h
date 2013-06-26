///////////////////////////////////////////////////////////////////////////////
// File: DoCastorAnalysis.h
// Date: 02.07 Panos Katsas
// Description: simulation analysis steering code 
//
///////////////////////////////////////////////////////////////////////////////
#undef debug
#ifndef DoCastorAnalysis_h
#define DoCastorAnalysis_h

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cassert>
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <memory>
#include <vector>

#include <CLHEP/Random/Randomize.h> 

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TLorentzVector.h"
#include "TUnixSystem.h"
#include "TSystem.h"
#include "TMath.h"
#include "TF1.h"

class G4Step;
class BeginOfJob;
class BeginOfRun;
class EndOfRun;
class BeginOfEvent;
class EndOfEvent;

class DoCastorAnalysis : public SimWatcher,
			 public Observer<const BeginOfJob *>, 
			 public Observer<const BeginOfRun *>,
			 public Observer<const EndOfRun *>,
			 public Observer<const BeginOfEvent *>, 
			 public Observer<const EndOfEvent *>,  
			 public Observer<const G4Step *> {  

public:
  DoCastorAnalysis(const edm::ParameterSet &p);
  virtual ~DoCastorAnalysis();

private:
  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const EndOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);
  void update(const G4Step * step);
  
private:

  int verbosity;

  std::string TreeFileName;

  TFile* CastorOutputEventFile;
  TTree* CastorTree;
  
  int eventIndex;

  std::vector<double> simhit_x, simhit_y, simhit_z;
  std::vector<double> simhit_eta, simhit_phi, simhit_energy;
  std::vector<int> simhit_sector, simhit_module;
  //std::vector<double> simhit_time;

  std::vector<double> *psimhit_x, *psimhit_y, *psimhit_z;
  std::vector<double> *psimhit_eta, *psimhit_phi,  *psimhit_energy;
  std::vector<int> *psimhit_sector, *psimhit_module;
  //std::vector<double> *psimhit_time;

  double simhit_etot;

};

#endif // DoCastorAnalysis_h

