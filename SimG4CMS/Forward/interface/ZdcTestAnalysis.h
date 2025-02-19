///////////////////////////////////////////////////////////////////////////////
// File: ZdcTestAnalysis.h
// Date: 03.06 Edmundo Garcia
// Description: simulation analysis steering code 
//
///////////////////////////////////////////////////////////////////////////////
#undef debug
#ifndef ZdcTestAnalysis_h
#define ZdcTestAnalysis_h

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

class ZdcTestAnalysis : public SimWatcher,
			public Observer<const BeginOfJob *>, 
			public Observer<const BeginOfRun *>,
			public Observer<const EndOfRun *>,
			public Observer<const BeginOfEvent *>, 
			public Observer<const EndOfEvent *>, 
			public Observer<const G4Step *> {
  
public:
  ZdcTestAnalysis(const edm::ParameterSet &p);
  virtual ~ZdcTestAnalysis();

private:
  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const EndOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);
  void update(const G4Step * step);
  
private:

  void   finish();

  int verbosity;
  int doNTzdcstep;
  int doNTzdcevent;
  std::string stepNtFileName;
  std::string eventNtFileName;

  TFile* zdcOutputEventFile;
  TFile* zdcOutputStepFile;

  TNtuple* zdcstepntuple;
  TNtuple* zdceventntuple;

  int eventIndex;
  int stepIndex;

  Float_t zdcsteparray[18];
  Float_t zdceventarray[16];

};

#endif // ZdcTestAnalysis_h
