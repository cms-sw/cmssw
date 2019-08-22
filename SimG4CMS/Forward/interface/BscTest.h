#ifndef BscTest_H
#define BscTest_H

// system include files
#include <vector>
#include <iostream>
#include <memory>
#include <string>
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// necessary objects:
#include "FWCore/Framework/interface/ESHandle.h"

//
//
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
//#include "SimG4Core/Watcher/interface/SimProducer.h"
//#include "SimG4Core/Watcher/interface/SimWatcherMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4CMS/Forward/interface/BscG4Hit.h"

#include "G4VTouchable.hh"
#include <map>
#include <cmath>
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Random/Randomize.h>
// #include <fstream>

// ----------------------------------------------------------------
// Includes needed for Root ntupling
//
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

#include <TObjArray.h>
#include <TObjString.h>
#include <TNamed.h>

class BscAnalysisHistManager : public TNamed {
public:
  BscAnalysisHistManager(const TString& managername);
  ~BscAnalysisHistManager() override;

  TH1F* GetHisto(Int_t Number);
  TH1F* GetHisto(const TObjString& histname);

  TH2F* GetHisto2(Int_t Number);
  TH2F* GetHisto2(const TObjString& histname);

  void WriteToFile(const TString& fOutputFile, const TString& fRecreateFile);

private:
  void BookHistos();
  void StoreWeights();
  void HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup);
  void HistInit(
      const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup);

  const char* fTypeTitle;
  TObjArray* fHistArray;
  TObjArray* fHistNamesArray;
};

class BscNumberingScheme;

class BeginOfJob;
class BeginOfRun;
class EndOfRun;
class BeginOfEvent;
class EndOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;

/*
class ObserveBeginOfRun : private Observer<const BeginOfRun *> 
{
public:
    ObserveBeginOfRun();
private:
    void update(const BeginOfRun * run);
};

class ObserveEndOfRun : private Observer<const EndOfRun *> 
{
public:
    ObserveEndOfRun();
private:
    void update(const EndOfRun * run);
};

class ObserveBeginOfEvent : private Observer<const BeginOfEvent *> 
{
public:
    ObserveBeginOfEvent();
private:
    void update(const BeginOfEvent * evt);
};

class ObserveEndOfEvent : private Observer<const EndOfEvent *> 
{
public:
    ObserveEndOfEvent();
private:
    void update(const EndOfEvent * evt);
    //    std::vector<BscG4Hit> theStripHits;
};

class ObserveBeginOfTrack : private Observer<const BeginOfTrack *> 
{
public:
    ObserveBeginOfTrack();
private:
    void update(const BeginOfTrack * trk);
};

class ObserveEndOfTrack : private Observer<const EndOfTrack *> 
{
public:
    ObserveEndOfTrack();
private:
    void update(const EndOfTrack * trk);
};

class ObserveStep : private Observer<const G4Step *> 
{
public:
    ObserveStep();
private:
    void update(const G4Step * step);
};
*/
//class BscTest: public SimProducer,
class BscTest : public SimWatcher,
                public Observer<const BeginOfJob*>,
                public Observer<const BeginOfRun*>,
                public Observer<const EndOfRun*>,
                public Observer<const BeginOfEvent*>,
                public Observer<const BeginOfTrack*>,
                public Observer<const G4Step*>,
                public Observer<const EndOfTrack*>,
                public Observer<const EndOfEvent*> {
public:
  BscTest(const edm::ParameterSet& p);
  ~BscTest() override;
  //MyActions();
  //MyActions();
private:
  // observer classes
  void update(const BeginOfJob* run) override;
  void update(const BeginOfRun* run) override;
  void update(const EndOfRun* run) override;
  void update(const BeginOfEvent* evt) override;
  void update(const BeginOfTrack* trk) override;
  void update(const G4Step* step) override;
  void update(const EndOfTrack* trk) override;
  void update(const EndOfEvent* evt) override;

private:
  //UHB_Analysis* UserNtuples;
  BscNumberingScheme* theBscNumberingScheme;

  int iev;
  int itrk;
  G4double entot0, tracklength0;

private:
  // Utilities to get detector levels during a step

  int detLevels(const G4VTouchable*) const;
  G4String detName(const G4VTouchable*, int, int) const;
  void detectorLevel(const G4VTouchable*, int&, int*, G4String*) const;

  double rinCalo, zinCalo;
  int lastTrackID;
  int verbosity;

  // SumEnerDeposit - all deposited energy on all steps ;  SumStepl - length in steel !!!
  G4double SumEnerDeposit, SumStepl, SumStepc;
  // numofpart - # particles produced along primary track
  int numofpart;
  // last point of the primary track
  G4ThreeVector lastpo;

  // z:
  double z1, z2, z3, z4;

private:
  Float_t bsceventarray[1];
  TNtuple* bsceventntuple;
  TFile bscOutputFile;
  int whichevent;

  BscAnalysisHistManager* TheHistManager;  //Histogram Manager of the analysis
  std::string fDataLabel;                  // Data type label
  std::string fOutputFile;                 // The output file name
  std::string fRecreateFile;               // Recreate the file flag, default="RECREATE"

  //  //  //  //  //  //  TObjString fHistType;
  //   TString fDataLabel;             // Data type label
  //   TString fOutputFile;            // The output file name
  //   TString fRecreateFile;          // Recreate the file flag, default="RECREATE"
};

#endif
