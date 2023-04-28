// system include files
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

//
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

// to retreive hits
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"

// G4 stuff
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "G4TransportationManager.hh"
#include "G4ProcessManager.hh"
#include "G4VTouchable.hh"

#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Random/Randomize.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

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
#include "TObjArray.h"
#include "TObjString.h"
#include "TNamed.h"

//#define EDM_ML_DEBUG

class BscAnalysisHistManager : public TNamed {
public:
  BscAnalysisHistManager(const TString& managername);
  ~BscAnalysisHistManager() override;

  TH1F* getHisto(Int_t Number);
  TH1F* getHisto(const TObjString& histname);

  TH2F* getHisto2(Int_t Number);
  TH2F* getHisto2(const TObjString& histname);

  void writeToFile(const TString& fOutputFile, const TString& fRecreateFile);

private:
  void bookHistos();
  void storeWeights();
  void histInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup);
  void histInit(
      const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup);

  const char* fTypeTitle;
  TObjArray* fHistArray;
  TObjArray* fHistNamesArray;
};

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

  //UHB_Analysis* UserNtuples;
  BscNumberingScheme* theBscNumberingScheme;

  int iev;
  int itrk;
  G4double entot0, tracklength0;

  // Utilities to get detector levels during a step

  int detLevels(const G4VTouchable*) const;
  G4String detName(const G4VTouchable*, int, int) const;
  void detectorLevel(const G4VTouchable*, int&, int*, G4String*) const;

  double rinCalo, zinCalo;
  int lastTrackID;
  int verbosity;

  // sumEnerDeposit - all deposited energy on all steps ;  sumStepl - length in steel !!!
  G4double sumEnerDeposit, sumStepl, sumStepc;
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
};

enum ntbsc_elements { ntbsc_evt };

//================================================================
BscTest::BscTest(const edm::ParameterSet& p) {
  //constructor
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("BscTest");
  verbosity = m_Anal.getParameter<int>("Verbosity");
  //verbosity    = 1;

  fDataLabel = m_Anal.getParameter<std::string>("FDataLabel");
  fOutputFile = m_Anal.getParameter<std::string>("FOutputFile");
  fRecreateFile = m_Anal.getParameter<std::string>("FRecreateFile");

  if (verbosity > 0) {
    edm::LogVerbatim("BscTest") << "============================================================================";
    edm::LogVerbatim("BscTest") << "BscTestconstructor :: Initialized as observer";
  }
  // Initialization:

  theBscNumberingScheme = new BscNumberingScheme();
  bsceventntuple = new TNtuple("NTbscevent", "NTbscevent", "evt");
  whichevent = 0;
  TheHistManager = new BscAnalysisHistManager(fDataLabel);

  if (verbosity > 0) {
    edm::LogVerbatim("BscTest") << "BscTest constructor :: Initialized BscAnalysisHistManager";
  }
}

BscTest::~BscTest() {
  //  delete UserNtuples;
  delete theBscNumberingScheme;

  TFile bscOutputFile("newntbsc.root", "RECREATE");
  edm::LogVerbatim("BscTest") << "Bsc output root file has been created";
  bsceventntuple->Write();
  edm::LogVerbatim("BscTest") << ", written";
  bscOutputFile.Close();
  edm::LogVerbatim("BscTest") << ", closed";
  delete bsceventntuple;
  edm::LogVerbatim("BscTest") << ", and deleted";

  //------->while end

  // Write histograms to file
  TheHistManager->writeToFile(fOutputFile, fRecreateFile);
  if (verbosity > 0) {
    edm::LogVerbatim("BscTest") << std::endl << "BscTest Destructor  -------->  End of BscTest : ";
  }

  edm::LogVerbatim("BscTest") << "BscTest: End of process";
}

//================================================================
// Histoes:
//-----------------------------------------------------------------------------

BscAnalysisHistManager::BscAnalysisHistManager(const TString& managername) {
  // The Constructor

  fTypeTitle = managername;
  fHistArray = new TObjArray();       // Array to store histos
  fHistNamesArray = new TObjArray();  // Array to store histos's names

  bookHistos();

  fHistArray->Compress();  // Removes empty space
  fHistNamesArray->Compress();
}
//-----------------------------------------------------------------------------

BscAnalysisHistManager::~BscAnalysisHistManager() {
  // The Destructor

  if (fHistArray) {
    fHistArray->Delete();
    delete fHistArray;
  }

  if (fHistNamesArray) {
    fHistNamesArray->Delete();
    delete fHistNamesArray;
  }
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::bookHistos() {
  // at Start: (mm)
  histInit("TrackPhi", "Primary Phi", 100, 0., 360.);
  histInit("TrackTheta", "Primary Theta", 100, 0., 180.);
  histInit("TrackP", "Track XY position Z+ ", 80, -80., 80., 80, -80., 80.);
  histInit("TrackM", "Track XY position Z-", 80, -80., 80., 80, -80., 80.);
  histInit("DetIDs", "Track DetId - vs +", 16, -0.5, 15.5, 16, 15.5, 31.5);
}

//-----------------------------------------------------------------------------

void BscAnalysisHistManager::writeToFile(const TString& fOutputFile, const TString& fRecreateFile) {
  //Write to file = fOutputFile

  edm::LogVerbatim("BscTest") << "================================================================";
  edm::LogVerbatim("BscTest") << " Write this Analysis to File " << fOutputFile;
  edm::LogVerbatim("BscTest") << "================================================================";

  TFile* file = new TFile(fOutputFile, fRecreateFile);

  fHistArray->Write();
  file->Close();
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::histInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup) {
  // Add histograms and histograms names to the array

  char* newtitle = new char[strlen(title) + strlen(fTypeTitle) + 5];
  strcpy(newtitle, title);
  strcat(newtitle, " (");
  strcat(newtitle, fTypeTitle);
  strcat(newtitle, ") ");
  fHistArray->AddLast((new TH1F(name, newtitle, nbinsx, xlow, xup)));
  fHistNamesArray->AddLast(new TObjString(name));
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::histInit(
    const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup) {
  // Add histograms and histograms names to the array

  char* newtitle = new char[strlen(title) + strlen(fTypeTitle) + 5];
  strcpy(newtitle, title);
  strcat(newtitle, " (");
  strcat(newtitle, fTypeTitle);
  strcat(newtitle, ") ");
  fHistArray->AddLast((new TH2F(name, newtitle, nbinsx, xlow, xup, nbinsy, ylow, yup)));
  fHistNamesArray->AddLast(new TObjString(name));
}
//-----------------------------------------------------------------------------

TH1F* BscAnalysisHistManager::getHisto(Int_t Number) {
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast() && fHistArray->At(Number) != (TObject*)nullptr) {
    return (TH1F*)(fHistArray->At(Number));

  } else {
    edm::LogVerbatim("BscTest") << "!!!!!!!!!!!!!!!!!!getHisto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    edm::LogVerbatim("BscTest") << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number;
    edm::LogVerbatim("BscTest") << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

    return (TH1F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH2F* BscAnalysisHistManager::getHisto2(Int_t Number) {
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast() && fHistArray->At(Number) != (TObject*)nullptr) {
    return (TH2F*)(fHistArray->At(Number));

  } else {
    edm::LogVerbatim("BscTest") << "!!!!!!!!!!!!!!!!getHisto2!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    edm::LogVerbatim("BscTest") << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number;
    edm::LogVerbatim("BscTest") << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

    return (TH2F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH1F* BscAnalysisHistManager::getHisto(const TObjString& histname) {
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return getHisto(index);
}
//-----------------------------------------------------------------------------

TH2F* BscAnalysisHistManager::getHisto2(const TObjString& histname) {
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return getHisto2(index);
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::storeWeights() {
  // Add structure to each histogram to store the weights

  for (int i = 0; i < fHistArray->GetEntries(); i++) {
    ((TH1F*)(fHistArray->At(i)))->Sumw2();
  }
}

//==================================================================== per JOB
void BscTest::update(const BeginOfJob* job) {
  //job
  edm::LogVerbatim("BscTest") << "BscTest:beggining of job";
  ;
}

//==================================================================== per RUN
void BscTest::update(const BeginOfRun* run) {
  //run

  edm::LogVerbatim("BscTest") << std::endl << "BscTest:: Begining of Run";
}

void BscTest::update(const EndOfRun* run) { ; }

//=================================================================== per EVENT
void BscTest::update(const BeginOfEvent* evt) {
  iev = (*evt)()->GetEventID();
  if (verbosity > 0) {
    edm::LogVerbatim("BscTest") << "BscTest:update Event number = " << iev;
  }
  whichevent++;
}

//=================================================================== per Track
void BscTest::update(const BeginOfTrack* trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    edm::LogVerbatim("BscTest") << "BscTest:update BeginOfTrack number = " << itrk;
  }
  if (itrk == 1) {
    sumEnerDeposit = 0.;
    numofpart = 0;
    sumStepl = 0.;
    sumStepc = 0.;
    tracklength0 = 0.;
  }
}

//=================================================================== per EndOfTrack
void BscTest::update(const EndOfTrack* trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    edm::LogVerbatim("BscTest") << "BscTest:update EndOfTrack number = " << itrk;
  }
  if (itrk == 1) {
    G4double tracklength = (*trk)()->GetTrackLength();  // Accumulated track length

    TheHistManager->getHisto("SumEDep")->Fill(sumEnerDeposit);
    TheHistManager->getHisto("TrackL")->Fill(tracklength);

    // direction !!!
    G4ThreeVector vert_mom = (*trk)()->GetVertexMomentumDirection();
    G4ThreeVector vert_pos = (*trk)()->GetVertexPosition();  // vertex ,where this track was created

    // last step information
    const G4Step* aStep = (*trk)()->GetStep();
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    lastpo = preStepPoint->GetPosition();
  }
}

// ====================================================

//=================================================================== each STEP
void BscTest::update(const G4Step* aStep) {
  // ==========================================================================

  if (verbosity > 2) {
    G4int stepnumber = aStep->GetTrack()->GetCurrentStepNumber();
    edm::LogVerbatim("BscTest") << "BscTest:update Step number = " << stepnumber;
  }
  // track on aStep:                                                                                         !
  G4Track* theTrack = aStep->GetTrack();
  TrackInformation* trkInfo = dynamic_cast<TrackInformation*>(theTrack->GetUserInformation());
  if (trkInfo == nullptr) {
    edm::LogVerbatim("BscTest") << "BscTest on aStep: No trk info !!!! abort ";
  }
  G4int id = theTrack->GetTrackID();
  G4String particleType = theTrack->GetDefinition()->GetParticleName();  //   !!!
  G4int parentID = theTrack->GetParentID();                              //   !!!
  G4TrackStatus trackstatus = theTrack->GetTrackStatus();                //   !!!
  G4double tracklength = theTrack->GetTrackLength();                     // Accumulated track length
  G4ThreeVector trackmom = theTrack->GetMomentum();
  G4double entot = theTrack->GetTotalEnergy();  //   !!! deposited on step
  G4int curstepnumber = theTrack->GetCurrentStepNumber();
  G4double stepl = aStep->GetStepLength();
  G4double EnerDeposit = aStep->GetTotalEnergyDeposit();
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4ThreeVector& preposition = preStepPoint->GetPosition();
  G4ThreeVector prelocalpoint = theTrack->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(preposition);
  G4VPhysicalVolume* currentPV = preStepPoint->GetPhysicalVolume();
  const G4String& prename = currentPV->GetName();

  const G4VTouchable* pre_touch = preStepPoint->GetTouchable();
  int pre_levels = detLevels(pre_touch);
  G4String name1[20];
  int copyno1[20];
  for (int i = 0; i < 20; ++i) {
    name1[i] = "";
    copyno1[i] = 0;
  }
  if (pre_levels > 0) {
    detectorLevel(pre_touch, pre_levels, copyno1, name1);
  }

  if (id == 1) {
    // on 1-st step:
    if (curstepnumber == 1) {
      entot0 = entot;
      //UserNtuples->fillg519(entot0,1.);
    }

    // on every step:

    // for Copper:
    if (prename == "SBST") {
      sumStepc += stepl;
      // =========
    }
    // for ststeel:
    //	 if(prename == "SBSTs") {
    if (prename == "SBSTs") {
      sumStepl += stepl;
      // =========
    }
    // =========
    // =========

    // exclude last track point if it is in SD (MI was started their)
    if (trackstatus != 2) {
      // for SD:   Si Det.:   SISTATION:SIPLANE:(SIDETL+BOUNDDET        +SIDETR + CERAMDET)
      if (prename == "SIDETL" || prename == "SIDETR") {
        if (prename == "SIDETL") {
          //UserNtuples->fillg569(EnerDeposit,1.);
        }
        if (prename == "SIDETR") {
          //UserNtuples->fillg570(EnerDeposit,1.);
        }

        G4String posname = aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName();
        if ((prename == "SIDETL" && posname != "SIDETL") || (prename == "SIDETR" && posname != "SIDETR")) {
          if (name1[2] == "SISTATION") {
            //UserNtuples->fillg539(copyno1[2],1.);
          }
          if (name1[3] == "SIPLANE") {
            //UserNtuples->fillg540(copyno1[3],1.);
          }

          if (prename == "SIDETL") {
            //UserNtuples->fillg541(EnerDeposit,1.);
            //UserNtuples->fillg561(numbcont,1.);
            if (copyno1[2] < 2) {
              //UserNtuples->fillg571(dx,1.);
            } else if (copyno1[2] < 3) {
              //UserNtuples->fillg563(dx,1.);
              if (copyno1[3] < 2) {
              } else if (copyno1[3] < 3) {
                //UserNtuples->fillg572(dx,1.);
              } else if (copyno1[3] < 4) {
                //UserNtuples->fillg573(dx,1.);
              } else if (copyno1[3] < 5) {
                //UserNtuples->fillg574(dx,1.);
              } else if (copyno1[3] < 6) {
                //UserNtuples->fillg575(dx,1.);
              } else if (copyno1[3] < 7) {
                //UserNtuples->fillg576(dx,1.);
              } else if (copyno1[3] < 8) {
                //UserNtuples->fillg577(dx,1.);
              } else if (copyno1[3] < 9) {
                //UserNtuples->fillg578(dx,1.);
              } else if (copyno1[3] < 10) {
                //UserNtuples->fillg579(dx,1.);
              }
            } else if (copyno1[2] < 4) {
              //UserNtuples->fillg565(dx,1.);
            } else if (copyno1[2] < 5) {
              //UserNtuples->fillg567(dx,1.);
            }
          }
          if (prename == "SIDETR") {
            //UserNtuples->fillg542(EnerDeposit,1.);
            //UserNtuples->fillg562(numbcont,1.);
            if (copyno1[2] < 2) {
              //UserNtuples->fillg581(dy,1.);
            } else if (copyno1[2] < 3) {
              //UserNtuples->fillg564(dy,1.);
              if (copyno1[3] < 2) {
              } else if (copyno1[3] < 3) {
                //UserNtuples->fillg582(dy,1.);
              } else if (copyno1[3] < 4) {
                //UserNtuples->fillg583(dy,1.);
              } else if (copyno1[3] < 5) {
                //UserNtuples->fillg584(dy,1.);
              } else if (copyno1[3] < 6) {
                //UserNtuples->fillg585(dy,1.);
              } else if (copyno1[3] < 7) {
                //UserNtuples->fillg586(dy,1.);
              } else if (copyno1[3] < 8) {
                //UserNtuples->fillg587(dy,1.);
              } else if (copyno1[3] < 9) {
                //UserNtuples->fillg588(dy,1.);
              } else if (copyno1[3] < 10) {
                //UserNtuples->fillg589(dy,1.);
              }
            } else if (copyno1[2] < 4) {
              //UserNtuples->fillg566(dy,1.);
            } else if (copyno1[2] < 5) {
              //UserNtuples->fillg568(dy,1.);
            }
          }
        }
      }
      // end of prenames SIDETL // SIDETR
    }
    // end of trackstatus != 2

    sumEnerDeposit += EnerDeposit;
    if (trackstatus == 2) {
      // primary track length
      //      //UserNtuples->fillg508(tracklength,1.);
      tracklength0 = tracklength;
    }
  }
  // end of primary track !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (parentID == 1 && curstepnumber == 1) {
    // particles deposit their energy along primary track
    numofpart += 1;
    if (prename == "SBST") {
      //UserNtuples->filld225(vert_pos.y(),vert_pos.z(),1.);
    }
    if (prename == "SBSTs") {
      //UserNtuples->filld226(vert_pos.y(),vert_pos.z(),1.);
    }
  }
}
// ==========================================================================
// ==========================================================================
int BscTest::detLevels(const G4VTouchable* touch) const {
  //Return number of levels
  if (touch)
    return ((touch->GetHistoryDepth()) + 1);
  else
    return 0;
}
// ==========================================================================

G4String BscTest::detName(const G4VTouchable* touch, int level, int currentlevel) const {
  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel;
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void BscTest::detectorLevel(const G4VTouchable* touch, int& level, int* copyno, G4String* name) const {
  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != nullptr)
        name[ii] = pv->GetName();
      else
        name[ii] = "Unknown";
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
// ==========================================================================

//===================================================================   End Of Event
void BscTest::update(const EndOfEvent* evt) {
  // ==========================================================================

  if (verbosity > 1) {
    iev = (*evt)()->GetEventID();
    edm::LogVerbatim("BscTest") << "BscTest:update EndOfEvent = " << iev;
  }
  // Fill-in ntuple
  bsceventarray[ntbsc_evt] = (float)whichevent;

  //
  int trackID = 0;
  G4PrimaryParticle* thePrim = nullptr;

  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  if (nvertex != 1)
    edm::LogVerbatim("BscTest") << "BscTest: My warning: NumberOfPrimaryVertex != 1  -->  = " << nvertex;

  for (int i = 0; i < nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == nullptr) {
      edm::LogVerbatim("BscTest") << "BscTest  End Of Event ERR: pointer to vertex = 0";
      continue;
    }
    G4int npart = avertex->GetNumberOfParticle();
    if (npart != 1)
      edm::LogVerbatim("BscTest") << "BscTest: My warning: NumberOfPrimaryPart != 1  -->  = " << npart;
    if (npart == 0)
      edm::LogVerbatim("BscTest") << "BscTest End Of Event ERR: no NumberOfParticle";

    if (thePrim == nullptr)
      thePrim = avertex->GetPrimary(trackID);

    if (thePrim != nullptr) {
      // primary vertex:
      G4double vx = 0., vy = 0., vz = 0.;
      vx = avertex->GetX0();
      vy = avertex->GetY0();
      vz = avertex->GetZ0();
      //UserNtuples->fillh01(vx);
      //UserNtuples->fillh02(vy);
      //UserNtuples->fillh03(vz);
      TheHistManager->getHisto("VtxX")->Fill(vx);
      TheHistManager->getHisto("VtxY")->Fill(vy);
      TheHistManager->getHisto("VtxZ")->Fill(vz);
    }
  }
  // prim.vertex loop end

  //=========================== thePrim != 0 ================================================================================
  if (thePrim != nullptr) {
    //
    // number of secondary particles deposited their energy along primary track
    //UserNtuples->fillg518(numofpart,1.);
    if (lastpo.z() < z4 && lastpo.perp() < 100.) {
      //UserNtuples->fillg536(numofpart,1.);
    }
    //

    // direction !!!
    G4ThreeVector mom = thePrim->GetMomentum();

    double phi = atan2(mom.y(), mom.x());
    if (phi < 0.)
      phi += twopi;
    double phigrad = phi * 180. / pi;

    double th = mom.theta();
    double eta = -log(tan(th / 2));
    TheHistManager->getHisto("PrimaryEta")->Fill(eta);
    TheHistManager->getHisto("PrimaryPhigrad")->Fill(phigrad);
    TheHistManager->getHisto("PrimaryTh")->Fill(th * 180. / pi);

    TheHistManager->getHisto("PrimaryLastpoZ")->Fill(lastpo.z());
    if (lastpo.z() < z4) {
      TheHistManager->getHisto("PrimaryLastpoX")->Fill(lastpo.x());
      TheHistManager->getHisto("PrimaryLastpoY")->Fill(lastpo.y());
    }
    if (numofpart > 4) {
      TheHistManager->getHisto("XLastpoNumofpart")->Fill(lastpo.x());
      TheHistManager->getHisto("YLastpoNumofpart")->Fill(lastpo.y());
    }

    // ==========================================================================

    // hit map for Bsc
    // ==================================

    std::map<int, float, std::less<int> > themap;
    std::map<int, float, std::less<int> > themap1;

    std::map<int, float, std::less<int> > themapxy;
    std::map<int, float, std::less<int> > themapz;
    // access to the G4 hit collections:  -----> this work OK:
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("BscTest") << "1";
#endif
    G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("BscTest") << "2";
#endif
    if (verbosity > 0) {
      edm::LogVerbatim("BscTest") << "BscTest:  accessed all HC";
      ;
    }
    int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("BSCHits");

    BscG4HitCollection* theCAFI = (BscG4HitCollection*)allHC->GetHC(CAFIid);
    if (verbosity > 0) {
      edm::LogVerbatim("BscTest") << "BscTest: theCAFI->entries = " << theCAFI->entries();
    }
    int varia;  // = 0 -all; =1 - MI; =2 - noMI
    //varia = 0;
    if (lastpo.z() < z4) {
      varia = 1;
    } else {
      varia = 2;
    }  // no MI end:
    int nhits = theCAFI->entries();
    for (int j = 0; j < nhits; j++) {
      BscG4Hit* aHit = (*theCAFI)[j];
      const CLHEP::Hep3Vector& hitPoint = aHit->getEntry();
      double zz = hitPoint.z();
      TheHistManager->getHisto("zHits")->Fill(zz);
      if (tracklength0 > 8300.)
        TheHistManager->getHisto("zHitsTrLoLe")->Fill(zz);
    }

    if (varia == 2) {
      //int nhit11 = 0, nhit12 = 0, nhit13 = 0;
      double totallosenergy = 0.;
      for (int j = 0; j < nhits; j++) {
        BscG4Hit* aHit = (*theCAFI)[j];

        const CLHEP::Hep3Vector& hitEntryLocalPoint = aHit->getEntryLocalP();
        const CLHEP::Hep3Vector& hitExitLocalPoint = aHit->getExitLocalP();
        const CLHEP::Hep3Vector& hitPoint = aHit->getEntry();
        //int trackIDhit = aHit->getTrackID();
        unsigned int unitID = aHit->getUnitID();
        double losenergy = aHit->getEnergyLoss();

        double zz = hitPoint.z();

        TheHistManager->getHisto("zHitsnoMI")->Fill(zz);

        if (verbosity > 2) {
          edm::LogVerbatim("BscTest") << "BscTest:zHits = " << zz;
        }

        themap[unitID] += losenergy;
        totallosenergy += losenergy;

        int zside;
        //int sector;
        BscNumberingScheme::unpackBscIndex(unitID);
        zside = (unitID & 32) >> 5;
        //sector = (unitID & 7);

        //
        //=======================================
        G4ThreeVector middle = (hitExitLocalPoint + hitEntryLocalPoint) / 2.;
        themapz[unitID] = hitPoint.z() + middle.z();
        //=======================================
        // Y
        if (zside == 1) {
          //UserNtuples->fillg24(losenergy,1.);
          if (losenergy > 0.00003) {
            themap1[unitID] += 1.;
          }
        }
        //X
        else if (zside == 2) {
          //UserNtuples->fillg25(losenergy,1.);
          if (losenergy > 0.00005) {
            themap1[unitID] += 1.;
          }
        }
        //	   }
        //
        /*
        if (sector == 1) {
          nhit11 += 1;
          //UserNtuples->fillg33(rr,1.);
          //UserNtuples->fillg11(yy,1.);
        }
        if (sector == 2) {
          nhit12 += 1;
          //UserNtuples->fillg34(rr,1.);
          //UserNtuples->fillg86(yy,1.);
        }
        if (sector == 3) {
          nhit13 += 1;
          //UserNtuples->fillg35(rr,1.);
          //UserNtuples->fillg87(yy,1.);
        }

        if (lastpo.z() < z4 && lastpo.perp() < 120.) {
          // MIonly:
          //UserNtuples->fillg16(lastpo.z(),1.);
          //UserNtuples->fillg18(zz,1.);
          //                                                                     Station I
          if (zz < z2) {
            //UserNtuples->fillg54(dx,1.);
            //UserNtuples->fillg55(dy,1.);
          }
          //                                                                     Station II
          if (zz < z3 && zz > z2) {
            //UserNtuples->fillg50(dx,1.);
            //UserNtuples->fillg51(dy,1.);
          }
          //                                                                     Station III
          if (zz < z4 && zz > z3) {
            //UserNtuples->fillg64(dx,1.);
            //UserNtuples->fillg65(dy,1.);
            //UserNtuples->filld209(xx,yy,1.);
          }
        } else {
          // no MIonly:
          //UserNtuples->fillg17(lastpo.z(),1.);
          //UserNtuples->fillg19(zz,1.);
          //UserNtuples->fillg74(incidentEnergyHit,1.);
          //UserNtuples->fillg75(float(trackIDhit),1.);
          //                                                                     Station I
          if (zz < z2) {
            //UserNtuples->fillg56(dx,1.);
            //UserNtuples->fillg57(dy,1.);
            //UserNtuples->fillg20(numofpart,1.);
            //UserNtuples->fillg21(sumEnerDeposit,1.);
            if (zside == 1) {
              //UserNtuples->fillg26(losenergy,1.);
            }
            if (zside == 2) {
              //UserNtuples->fillg76(losenergy,1.);
            }
            if (trackIDhit == 1) {
              //UserNtuples->fillg70(dx,1.);
              //UserNtuples->fillg71(incidentEnergyHit,1.);
              //UserNtuples->fillg79(losenergy,1.);
            } else {
              //UserNtuples->fillg82(dx,1.);
            }
          }
          //                                                                     Station II
          if (zz < z3 && zz > z2) {
            //UserNtuples->fillg52(dx,1.);
            //UserNtuples->fillg53(dy,1.);
            //UserNtuples->fillg22(numofpart,1.);
            //UserNtuples->fillg23(sumEnerDeposit,1.);
            //UserNtuples->fillg80(incidentEnergyHit,1.);
            //UserNtuples->fillg81(float(trackIDhit),1.);
            if (zside == 1) {
              //UserNtuples->fillg27(losenergy,1.);
            }
            if (zside == 2) {
              //UserNtuples->fillg77(losenergy,1.);
            }
            if (trackIDhit == 1) {
              //UserNtuples->fillg72(dx,1.);
              //UserNtuples->fillg73(incidentEnergyHit,1.);
            } else {
              //UserNtuples->fillg83(dx,1.);
            }
          }
          //                                                                     Station III
          if (zz < z4 && zz > z3) {
            if (zside == 1) {
              //UserNtuples->fillg28(losenergy,1.);
            }
            if (zside == 2) {
              //UserNtuples->fillg78(losenergy,1.);
            }
          }
        }
        */
      }  // MIonly or noMIonly ENDED
      if (totallosenergy == 0.0) {
        edm::LogVerbatim("BscTest") << "BscTest:     number of hits = " << theCAFI->entries();
        for (int j = 0; j < nhits; j++) {
          BscG4Hit* aHit = (*theCAFI)[j];
          double losenergy = aHit->getEnergyLoss();
          edm::LogVerbatim("BscTest") << " j hits = " << j << "losenergy = " << losenergy;
        }
      }
      //   FIBRE Hit collected analysis
      /*
      double totalEnergy = 0.;
      int nhitsX = 0, nhitsY = 0, nsumhit = 0;
      for (int sector = 1; sector < 4; sector++) {
        int nhitsecX = 0, nhitsecY = 0;
        for (int zmodule = 1; zmodule < 11; zmodule++) {
          for (int zside = 1; zside < 3; zside++) {
            int det = 1;  // nhit = 0;
            //	int sScale = 20;
            int index = BscNumberingScheme::packBscIndex(det, zside, sector);
            double theTotalEnergy = themap[index];
            //   X planes
            if (zside < 2) {
              //UserNtuples->fillg47(theTotalEnergy,1.);
              if (theTotalEnergy > 0.00003) {
                nhitsX += 1;
                //		nhitsecX += themap1[index];
                //		nhit=1;
              }
            }
            //   Y planes
            else {
              //UserNtuples->fillg49(theTotalEnergy,1.);
              if (theTotalEnergy > 0.00005) {
                nhitsY += 1;
                //		nhitsecY += themap1[index];
                //		nhit=1;
              }
            }

            totalEnergy += themap[index];
          }  // for
        }    // for
             //UserNtuples->fillg39(nhitsecY,1.);
        if (nhitsecX > 10 || nhitsecY > 10) {
          nsumhit += 1;
          //UserNtuples->fillp213(float(sector),float(1.),1.);
        } else {  //UserNtuples->fillp213(float(sector),float(0.),1.);
        }
      }  // for

      if (nsumhit >= 2) {  //UserNtuples->fillp212(vy,float(1.),1.);
      } else {             //UserNtuples->fillp212(vy,float(0.),1.);
      }
      */
    }  // MI or no MI or all  - end
  }    // primary end

  if (verbosity > 0) {
    edm::LogVerbatim("BscTest") << "BscTest:  END OF Event " << (*evt)()->GetEventID();
  }
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(BscTest);
