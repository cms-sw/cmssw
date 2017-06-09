///////////////////////////////////////////////////////////////////////////////
// File: CastorTestAnalysis.h
// Date: 02.07 Panos Katsas
// Description: simulation analysis steering code 
//
///////////////////////////////////////////////////////////////////////////////
#undef debug
#ifndef CastorTestAnalysis_h
#define CastorTestAnalysis_h

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
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"

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
class CastorNumberingScheme;

class CastorTestAnalysis : public SimWatcher,
			public Observer<const BeginOfJob *>, 
			public Observer<const BeginOfRun *>,
			public Observer<const EndOfRun *>,
			public Observer<const BeginOfEvent *>, 
			public Observer<const EndOfEvent *>, 
			public Observer<const G4Step *> {
  
public:
  CastorTestAnalysis(const edm::ParameterSet &p);
  virtual ~CastorTestAnalysis();

private:
  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const EndOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);
  void update(const G4Step * step);
  
private:

  void getCastorBranchData(const CaloG4HitCollection * hc);
  void Finish();

  int verbosity;
  int doNTcastorstep;
  int doNTcastorevent;
  std::string stepNtFileName;
  std::string eventNtFileName;

  TFile* castorOutputEventFile;
  TFile* castorOutputStepFile;

  TNtuple* castorstepntuple;
  TNtuple* castoreventntuple;
  
  CastorNumberingScheme* theCastorNumScheme;

  int eventIndex;
  int stepIndex;
  int eventGlobalHit;

  Float_t castorsteparray[14];
  Float_t castoreventarray[11];

};

#endif // CastorTestAnalysis_h
