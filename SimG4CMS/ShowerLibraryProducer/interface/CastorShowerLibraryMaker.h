///////////////////////////////////////////////////////////////////////////////
//
// File: CastorShowerLibraryMaker.h
// Date: 02/2009 
// Author: Wagner Carvalho (adapted from Panos Katsas code)
// Description: simulation analysis steering code 
//
///////////////////////////////////////////////////////////////////////////////
#undef debug
#ifndef CastorShowerLibraryMaker_h
#define CastorShowerLibraryMaker_h

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// Classes for shower library Root file
#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"

#include <cassert>
#include <iostream>
#include <string>
#include <map>
#include <set>
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

/*
typedef std::vector<std::vector<CastorShowerEvent> > phi_t; //holds N phi bin collection (with M events each)
typedef std::vector<phi_t>                           eta_t; //holds N eta bin collection  
typedef std::vector<eta_t>                           energy_t; //holds N energy bin
*/
typedef std::vector<std::vector<std::vector<std::vector<CastorShowerEvent> > > > SLBin3D; // bin in energy, eta and phi

class CastorShowerLibraryMaker : public SimWatcher,
			public Observer<const BeginOfJob *>, 
			public Observer<const BeginOfRun *>,
			public Observer<const EndOfRun *>,
			public Observer<const BeginOfEvent *>, 
			public Observer<const EndOfEvent *>, 
			public Observer<const G4Step *> {
  
public:

  CastorShowerLibraryMaker(const edm::ParameterSet &p);
  virtual ~CastorShowerLibraryMaker();

private:
  typedef int ebin;
  typedef int etabin;
  typedef int phibin;
// private structures
  struct ShowerLib {
         CastorShowerLibraryInfo SLInfo; // the info
         SLBin3D                 SLCollection; // the showers
         std::vector<double>     SLEnergyBins;
         std::vector<double>     SLEtaBins;
         std::vector<double>     SLPhiBins;
         unsigned int            nEvtPerBinE;
         unsigned int            nEvtPerBinEta;
         unsigned int            nEvtPerBinPhi;
         std::vector<int>                             nEvtInBinE;
         std::vector<std::vector<int> >               nEvtInBinEta;
         std::vector<std::vector<std::vector<int> > > nEvtInBinPhi;
  }; 

  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const EndOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);
  void update(const G4Step * step);

private:

  void   Finish();

  // Job general parameters
  int verbosity;
  std::string eventNtFileName;
  
  unsigned int NPGParticle; // number of particles requested to Particle Gun
  std::vector<int> PGParticleIDs; //p. gun particle IDs
  bool DoHadSL; // true if hadronic SL should be produced
  bool DoEmSL;  // true if electromag. SL should be produced
  
  // Pointers for user defined class objects to be stored to Root file
  CastorShowerLibraryInfo   *emInfo;
  CastorShowerLibraryInfo   *hadInfo;
  CastorShowerEvent     *emShower;
  CastorShowerEvent    *hadShower;
  ShowerLib            emSLHolder;
  ShowerLib            hadSLHolder;
  ShowerLib*           SLShowerptr; // pointer to the current shower collection (above)
  std::map<int,std::set<int> > MapOfSecondaries; // map to hold all secondaries ID keyed by
                                                 // the PDG code of the primary

// private methods
  int FindEnergyBin(double e);
  int FindEtaBin(double eta);
  int FindPhiBin(double phi);
  bool SLacceptEvent(int, int, int);
  bool IsSLReady();
  void GetKinematics(G4PrimaryParticle* ,
       double& px, double& py, double& pz, double& pInit, double& eta, double& phi);

  std::vector<G4PrimaryParticle*>  GetPrimary(const EndOfEvent * );
  bool FillShowerEvent(G4HCofThisEvent* ,CastorShowerEvent*, int);
  void InitSLHolder(ShowerLib& );

  void printSLstatus(int , int, int);
  int& SLnEvtInBinE(int ebin);
  int& SLnEvtInBinEta(int ebin, int etabin);
  int& SLnEvtInBinPhi(int ebin, int etabin, int phibin);
  bool         SLisEBinFilled(int ebin);
  bool         SLisEtaBinFilled(int ebin, int etabin);
  bool         SLisPhiBinFilled(int ebin, int etabin, int phibin);

  // Root pointers
  TFile* theFile;
  TTree* theTree;

  int eventIndex;
  int stepIndex;   // ignore, please

};

#endif // CastorShowerLibraryMaker_h
