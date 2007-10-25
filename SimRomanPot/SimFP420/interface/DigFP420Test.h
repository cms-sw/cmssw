#ifndef DigFP420Test_H
#define DigFP420Test_H

// system include files
#include<vector>
#include <iostream>
#include <memory>
#include <string>
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// necessary objects:
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
//
#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

//#include "SimRomanPot/SimFP420/interface/ClusterFP420.h"

// ----------------------------------------------------------------

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

// ----------------------------------------------------------------
#include "G4VTouchable.hh"
#include <map>
#include <cmath>
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Random/Randomize.h> 
// #include <fstream>
//using namespace std;


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

class Fp420AnalysisHistManager : public TNamed {
        public:

                Fp420AnalysisHistManager(TString managername);
                ~Fp420AnalysisHistManager();

                TH1F* GetHisto(Int_t Number);
                TH1F* GetHisto(const TObjString histname);

                TH2F* GetHisto2(Int_t Number);
                TH2F* GetHisto2(const TObjString histname);

                void WriteToFile(TString fOutputFile,TString fRecreateFile);

        private:

                void BookHistos();
                void StoreWeights();
                void HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup);
                void HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup);

                const char* fTypeTitle;
                TObjArray* fHistArray;
                TObjArray* fHistNamesArray;

};


class FP420NumberingScheme;

class HDigiFP420;
class DigitizerFP420;
class DigiCollectionFP420;

//class ClusterFP420;

class BeginOfRun;
class EndOfRun;
class BeginOfEvent;
class EndOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;


class DigFP420Test : public SimWatcher,
  public Observer<const BeginOfJob *>, 
  public Observer<const BeginOfRun *>,
  public Observer<const EndOfRun *>,
  public Observer<const BeginOfEvent *>,
  public Observer<const BeginOfTrack *>,
  public Observer<const G4Step *>,
  public Observer<const EndOfTrack *>,
  public Observer<const EndOfEvent *>
{
public:
  DigFP420Test(const edm::ParameterSet &conf);
  virtual ~DigFP420Test();
private:

  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const EndOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const BeginOfTrack * trk);
  void update(const G4Step * step);
  void update(const EndOfTrack * trk);
  void update(const EndOfEvent * evt);

private:
    edm::ParameterSet conf_;

  //UHB_Analysis* UserNtuples;
  FP420NumberingScheme * theFP420NumberingScheme;
  //  FP420DigiMain* theFP420DigiMain;
  DigitizerFP420* theDigitizerFP420;
  DigiCollectionFP420 output;

  std::vector<HDigiFP420> collector;

  int iev;
  int itrk;
  G4double entot0, tracklength0;

private:
// Utilities to get detector levels during a step

  int      detLevels(const G4VTouchable*) const;
  G4String  detName(const G4VTouchable*, int, int) const;
  void     detectorLevel(const G4VTouchable*, int&, int*, G4String*) const;


 double rinCalo, zinCalo;
 int    lastTrackID;
 int verbosity;

 // SumEnerDeposit - all deposited energy on all steps ;  SumStepl - length in steel !!!
 G4double      SumEnerDeposit, SumStepl, SumStepc;
 // numofpart - # particles produced along primary track
 int          numofpart;
 // last point of the primary track
 G4ThreeVector  lastpo;


 // z:
 //double z1, z2, z3, z4, zUnit; 
 double z1, z2, z3, z4, zD2, zD3; 
 // Number of Stations:
 int sn0;
 // Number of planes:
 int pn0;
 int xytype;
 // shift of planes:
 bool UseHalfPitchShiftInY_, UseThirdPitchShiftInY_, UseForthPitchShiftInY_;
 bool UseHalfPitchShiftInX_, UseThirdPitchShiftInX_, UseForthPitchShiftInX_;

 // detector:
 double ldriftX, ldriftY;
 double pitchX, pitchY;
 int numStripsX,numStripsY;
 int numStripsXW, numStripsYW;

 double	ZSiDetL, ZSiDetR, z420, zinibeg;
 double	ZGapLDet, ZBoundDet, ZSiStep, ZSiPlane;
 // double	zBlade, gapBlade, ZKapton, ZSiElectr, ZCeramDet;

 int ENC                ;
 float Thick300, dYYconst, dXXconst;
 double ElectronPerADC;
private:

  Float_t fp420eventarray[1];
  TNtuple* fp420eventntuple;
  TFile fp420OutputFile;
  int whichevent;

  Fp420AnalysisHistManager* TheHistManager;  //Histogram Manager of the analysis
  std::string fDataLabel;             // Data type label
  std::string fOutputFile;            // The output file name
  std::string fRecreateFile;          // Recreate the file flag, default="RECREATE"
};

#endif









