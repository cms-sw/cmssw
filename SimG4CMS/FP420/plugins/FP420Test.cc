// -*- C++ -*-
//

// system include files
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
//
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// to retreive hits
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420Test.h"

// G4 stuff
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "G4TransportationManager.hh"
#include "G4ProcessManager.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//================================================================
// Root stuff

// Include the standard header <cassert> to effectively include
// the standard header <assert.h> within the std namespace.
#include <cassert>

using namespace edm;
using namespace std;
//================================================================

//UserVerbosity FP420Test::std::cout("FP420Test","info","FP420Test");

enum ntfp420_elements { ntfp420_evt };
//================================================================
FP420Test::FP420Test(const edm::ParameterSet& p) {
  //constructor
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("FP420Test");
  verbosity = m_Anal.getParameter<int>("Verbosity");
  //verbosity    = 1;

  fDataLabel = m_Anal.getParameter<std::string>("FDataLabel");
  fOutputFile = m_Anal.getParameter<std::string>("FOutputFile");
  fRecreateFile = m_Anal.getParameter<std::string>("FRecreateFile");

  if (verbosity > 0) {
    std::cout << "============================================================================" << std::endl;
    std::cout << "FP420Testconstructor :: Initialized as observer" << std::endl;
  }
  // Initialization:

  pn0 = 6;
  sn0 = 3;
  rn00 = 7;

  z420 = 420000.0;  // mm
  zD2 = 4000.0;     // mm
  zD3 = 8000.0;     // mm
  //
  zBlade = 5.00;
  gapBlade = 1.6;
  double gapSupplane = 1.6;
  ZSiPlane = 2 * zBlade + gapBlade + gapSupplane;

  double ZKapton = 0.1;
  ZSiStep = ZSiPlane + ZKapton;

  double ZBoundDet = 0.020;
  double ZSiElectr = 0.250;
  double ZCeramDet = 0.500;
  //
  ZSiDet = 0.250;
  ZGapLDet = zBlade / 2 - (ZSiDet + ZSiElectr + ZBoundDet + ZCeramDet / 2);
  //
  //  ZSiStation = 5*(2*(5.+1.6)+0.1)+2*6.+1.0 =  79.5
  double ZSiStation = (pn0 - 1) * (2 * (zBlade + gapBlade) + ZKapton) + 2 * 6. + 0.0;  // =  78.5
  // 11.=e1, 12.=e2 in zzzrectangle.xml
  double eee1 = 11.;
  double eee2 = 12.;

  zinibeg = (eee1 - eee2) / 2.;

  z1 = zinibeg + (ZSiStation + 10.) / 2 + z420;  // z1 - right after 1st Station
  z2 = z1 + zD2;                                 //z2 - right after middle Station
  z3 = z1 + zD3;                                 //z3 - right after last   Station
  z4 = z1 + 2 * zD3;
  //==================================

  fp420eventntuple = new TNtuple("NTfp420event", "NTfp420event", "evt");

  whichevent = 0;

  //   fDataLabel      = "defaultData";
  //       fOutputFile     = "TheAnlysis.root";
  //       fRecreateFile   = "RECREATE";

  TheHistManager = new Fp420AnalysisHistManager(fDataLabel);

  if (verbosity > 0) {
    std::cout << "FP420Test constructor :: Initialized Fp420AnalysisHistManager" << std::endl;
  }
}

FP420Test::~FP420Test() {
  //  delete UserNtuples;

  TFile fp420OutputFile("newntfp420.root", "RECREATE");
  std::cout << "FP420 output root file has been created";
  fp420eventntuple->Write();
  std::cout << ", written";
  fp420OutputFile.Close();
  std::cout << ", closed";
  delete fp420eventntuple;
  std::cout << ", and deleted" << std::endl;

  //------->while end

  // Write histograms to file
  TheHistManager->WriteToFile(fOutputFile, fRecreateFile);
  if (verbosity > 0) {
    std::cout << std::endl << "FP420Test Destructor  -------->  End of FP420Test : " << std::endl;
  }
}

//================================================================
// Histoes:
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::Fp420AnalysisHistManager(const TString& managername) {
  // The Constructor

  fTypeTitle = managername;
  fHistArray = new TObjArray();       // Array to store histos
  fHistNamesArray = new TObjArray();  // Array to store histos's names

  TH1::AddDirectory(kFALSE);
  BookHistos();

  fHistArray->Compress();  // Removes empty space
  fHistNamesArray->Compress();

  //      StoreWeights();                    // Store the weights
}
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::~Fp420AnalysisHistManager() {
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

void Fp420AnalysisHistManager::BookHistos() {
  // at Start: (mm)
  HistInit("PrimaryEta", "Primary Eta", 100, 9., 12.);
  HistInit("PrimaryPhigrad", "Primary Phigrad", 100, 0., 360.);
  HistInit("PrimaryTh", "Primary Th", 100, 0., 180.);
  HistInit("PrimaryLastpoZ", "Primary Lastpo Z", 100, -200., 430000.);
  HistInit("PrimaryLastpoX", "Primary Lastpo X Z<z4", 100, -30., 30.);
  HistInit("PrimaryLastpoY", "Primary Lastpo Y Z<z4", 100, -30., 30.);
  HistInit("XLastpoNumofpart", "Primary Lastpo X n>10", 100, -30., 30.);
  HistInit("YLastpoNumofpart", "Primary Lastpo Y n>10", 100, -30., 30.);
  HistInit("VtxX", "Vtx X", 100, -50., 50.);
  HistInit("VtxY", "Vtx Y", 100, -50., 50.);
  HistInit("VtxZ", "Vtx Z", 100, -200., 430000.);
  // Book the histograms and add them to the array
  HistInit("SumEDep", "This is sum Energy deposited", 100, -1, 199.);
  HistInit("TrackL", "This is TrackL", 100, 0., 12000.);
  HistInit("zHits", "z Hits all events", 100, 400000., 430000.);
  HistInit("zHitsnoMI", "z Hits no MI", 100, 400000., 430000.);
  HistInit("zHitsTrLoLe", "z Hits TrLength bigger 8300", 100, 400000., 430000.);
  HistInit("NumberOfHits", "NumberOfHits", 100, 0., 300.);
}

//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::WriteToFile(const TString& fOutputFile, const TString& fRecreateFile) {
  //Write to file = fOutputFile

  std::cout << "================================================================" << std::endl;
  std::cout << " Write this Analysis to File " << fOutputFile << std::endl;
  std::cout << "================================================================" << std::endl;

  TFile* file = new TFile(fOutputFile, fRecreateFile);

  fHistArray->Write();
  file->Close();
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup) {
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

void Fp420AnalysisHistManager::HistInit(
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

TH1F* Fp420AnalysisHistManager::GetHisto(Int_t Number) {
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast() && fHistArray->At(Number) != (TObject*)nullptr) {
    return (TH1F*)(fHistArray->At(Number));

  } else {
    std::cout << "!!!!!!!!!!!!!!!!!!GetHisto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    return (TH1F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH2F* Fp420AnalysisHistManager::GetHisto2(Int_t Number) {
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast() && fHistArray->At(Number) != (TObject*)nullptr) {
    return (TH2F*)(fHistArray->At(Number));

  } else {
    std::cout << "!!!!!!!!!!!!!!!!GetHisto2!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    return (TH2F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH1F* Fp420AnalysisHistManager::GetHisto(const TObjString& histname) {
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return GetHisto(index);
}
//-----------------------------------------------------------------------------

TH2F* Fp420AnalysisHistManager::GetHisto2(const TObjString& histname) {
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return GetHisto2(index);
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::StoreWeights() {
  // Add structure to each histogram to store the weights

  for (int i = 0; i < fHistArray->GetEntries(); i++) {
    ((TH1F*)(fHistArray->At(i)))->Sumw2();
  }
}

// Histoes end :

//================================================================

// using several observers

//==================================================================== per JOB
void FP420Test::update(const BeginOfJob* job) {
  //job
  std::cout << "FP420Test:beggining of job" << std::endl;
  ;
}

//==================================================================== per RUN
void FP420Test::update(const BeginOfRun* run) {
  //run

  std::cout << std::endl << "FP420Test:: Begining of Run" << std::endl;
}

void FP420Test::update(const EndOfRun* run) { ; }

//=================================================================== per EVENT
void FP420Test::update(const BeginOfEvent* evt) {
  iev = (*evt)()->GetEventID();
  if (verbosity > 0) {
    std::cout << "FP420Test:update Event number = " << iev << std::endl;
  }
  whichevent++;
}

//=================================================================== per Track
void FP420Test::update(const BeginOfTrack* trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    std::cout << "FP420Test:update BeginOfTrack number = " << itrk << std::endl;
  }
  if (itrk == 1) {
    SumEnerDeposit = 0.;
    numofpart = 0;
    SumStepl = 0.;
    SumStepc = 0.;
    tracklength0 = 0.;
  }
}

//=================================================================== per EndOfTrack
void FP420Test::update(const EndOfTrack* trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    std::cout << "FP420Test:update EndOfTrack number = " << itrk << std::endl;
  }
  if (itrk == 1) {
    G4double tracklength = (*trk)()->GetTrackLength();  // Accumulated track length

    TheHistManager->GetHisto("SumEDep")->Fill(SumEnerDeposit);
    TheHistManager->GetHisto("TrackL")->Fill(tracklength);

    // direction !!!
    G4ThreeVector vert_mom = (*trk)()->GetVertexMomentumDirection();
    G4ThreeVector vert_pos = (*trk)()->GetVertexPosition();  // vertex ,where this track was created

    //    float eta = 0.5 * log( (1.+vert_mom.z()) / (1.-vert_mom.z()) );
    //    float phi = atan2(vert_mom.y(),vert_mom.x());
    //    if (phi < 0.) phi += twopi;
    //    float phigrad = phi*180./pi;

    //      float XV = vert_pos.x(); // mm
    //      float YV = vert_pos.y(); // mm
    //UserNtuples->fillg543(phigrad,1.);
    //UserNtuples->fillp203(phigrad,SumStepl,1.);
    //UserNtuples->fillg544(XV,1.);
    //UserNtuples->fillp201(XV,SumStepl,1.);
    //UserNtuples->fillg545(YV,1.);
    //UserNtuples->fillp202(YV,SumStepl,1.);

    //UserNtuples->fillg524(eta,1.);

    //UserNtuples->fillg534(SumStepl,1.);
    //UserNtuples->fillg535(eta,SumStepl);
    //UserNtuples->fillp207(eta,SumStepl,1.);
    //UserNtuples->filld217(eta,SumStepl,1.);
    //UserNtuples->filld220(phigrad,SumStepl,1.);
    //UserNtuples->filld221(XV,SumStepl,1.);
    //UserNtuples->filld222(YV,SumStepl,1.);
    //UserNtuples->fillg537(SumStepc,1.);
    //UserNtuples->fillg84(SumStepl,1.);

    // MI = (multiple interactions):
    if (tracklength < z4) {
      //        //UserNtuples->fillp208(eta,tracklength,1.);
      //UserNtuples->filld218(eta,tracklength,1.);
      //UserNtuples->fillg538(SumStepc,1.);
      //UserNtuples->fillg85(SumStepl,1.);
    }

    // last step information
    const G4Step* aStep = (*trk)()->GetStep();
    //   G4int csn = (*trk)()->GetCurrentStepNumber();
    //   G4double sl = (*trk)()->GetStepLength();
    // preStep
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    lastpo = preStepPoint->GetPosition();

    // Analysis:
    if (lastpo.z() < z1 && lastpo.perp() < 100.) {
      //UserNtuples->fillg525(eta,1.);
      //UserNtuples->fillg546(XV,1.);
      //UserNtuples->fillg551(YV,1.);
      //UserNtuples->fillg556(phigrad,1.);
    }
    if ((lastpo.z() > z1 && lastpo.z() < z2) && lastpo.perp() < 100.) {
      //UserNtuples->fillg526(eta,1.);
      //UserNtuples->fillg547(XV,1.);
      //UserNtuples->fillg552(YV,1.);
      //UserNtuples->fillg557(phigrad,1.);
    }
    if (lastpo.z() < z2 && lastpo.perp() < 100.) {
      //UserNtuples->fillg527(eta,1.);
      //UserNtuples->fillg548(XV,1.);
      //UserNtuples->fillg553(YV,1.);
      //UserNtuples->fillg558(phigrad,1.);
      //UserNtuples->fillg521(lastpo.x(),1.);
      //UserNtuples->fillg522(lastpo.y(),1.);
      //UserNtuples->fillg523(lastpo.z(),1.);
    }
    if (lastpo.z() < z3 && lastpo.perp() < 100.) {
      //UserNtuples->fillg528(eta,1.);
      //UserNtuples->fillg549(XV,1.);
      //UserNtuples->fillg554(YV,1.);
      //UserNtuples->fillg559(phigrad,1.);
    }
    if (lastpo.z() < z4 && lastpo.perp() < 100.) {
      //UserNtuples->fillg529(eta,1.);
      //UserNtuples->fillg550(XV,1.);
      //UserNtuples->fillg555(YV,1.);
      //UserNtuples->fillg560(phigrad,1.);
      //UserNtuples->fillg531(lastpo.x(),1.);
      //UserNtuples->fillg532(lastpo.y(),1.);
      //UserNtuples->fillg533(lastpo.z(),1.);
    }
  }
}

// ====================================================

//=================================================================== each STEP
void FP420Test::update(const G4Step* aStep) {
  // ==========================================================================

  if (verbosity > 2) {
    G4int stepnumber = aStep->GetTrack()->GetCurrentStepNumber();
    std::cout << "FP420Test:update Step number = " << stepnumber << std::endl;
  }
  // track on aStep:                                                                                         !
  G4Track* theTrack = aStep->GetTrack();
  TrackInformation* trkInfo = dynamic_cast<TrackInformation*>(theTrack->GetUserInformation());
  if (trkInfo == nullptr) {
    std::cout << "FP420Test on aStep: No trk info !!!! abort " << std::endl;
  }
  G4int id = theTrack->GetTrackID();
  G4String particleType = theTrack->GetDefinition()->GetParticleName();  //   !!!
  G4int parentID = theTrack->GetParentID();                              //   !!!
  G4TrackStatus trackstatus = theTrack->GetTrackStatus();                //   !!!
  G4double tracklength = theTrack->GetTrackLength();                     // Accumulated track length
  G4ThreeVector trackmom = theTrack->GetMomentum();
  G4double entot = theTrack->GetTotalEnergy();  //   !!! deposited on step
  G4int curstepnumber = theTrack->GetCurrentStepNumber();
  // const G4ThreeVector&   vert_pos       = theTrack->GetVertexPosition(); // vertex ,where this track was created
  // const G4ThreeVector&   vert_mom       = theTrack->GetVertexMomentumDirection();

  //   double costheta =vert_mom.z()/sqrt(vert_mom.x()*vert_mom.x()+vert_mom.y()*vert_mom.y()+vert_mom.z()*vert_mom.z());
  //   double theta = acos(min(max(costheta,double(-1.)),double(1.)));
  //  float eta = -log(tan(theta/2));
  //   double phi = -1000.;
  //   if (vert_mom.x() != 0) phi = atan2(vert_mom.y(),vert_mom.x());
  //   if (phi < 0.) phi += twopi;
  //   double phigrad = phi*360./twopi;

  //G4double       trackvel       = theTrack->GetVelocity();

  //std::cout << " trackstatus= " << trackstatus << " entot= " << entot  << std::endl;

  // step points:                                                                                         !
  G4double stepl = aStep->GetStepLength();
  G4double EnerDeposit = aStep->GetTotalEnergyDeposit();
  // pointers:                                                                                         !
  //G4VPhysicalVolume*  physvol       = theTrack->GetVolume();
  //G4VPhysicalVolume*  nextphysvol   = theTrack->GetNextVolume();
  //G4Material*       materialtr     = theTrack->GetMaterial();
  //G4Material*       nextmaterialtr = theTrack->GetNextMaterial();

  // preStep
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

  //  G4LogicalVolume*   lv            = currentPV->GetLogicalVolume();
  //  G4Material*       mat           = lv->GetMaterial();
  //  std::string prenameVolume;
  //  prenameVolume.assign(prename,0,20);

  //   G4double         prebeta          = preStepPoint->GetBeta();
  //   G4double         precharge        = preStepPoint->GetCharge();
  //  G4double          prerad           = mat->GetRadlen();

  //  std::cout << " EneryDeposited = " << EnerDeposit << std::endl;
  //  std::cout << " prevolume = "      << prename << std::endl;
  ////  std::cout << " posvolume = "      << aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName() << std::endl;
  //  std::cout << " preposition = "    << preposition << std::endl;
  /*
  // postStep
  G4StepPoint*      postStepPoint  = aStep->GetPostStepPoint();   
  G4ThreeVector     posposition    = postStepPoint->GetPosition();	
  G4ThreeVector     poslocalpoint  = theTrack->GetTouchable()->GetHistory()->
                                           GetTopTransform().TransformPoint(posposition);
  G4VPhysicalVolume* poscurrentPV      = postStepPoint->GetPhysicalVolume();
  G4String         posname        = poscurrentPV->GetName();
//  G4LogicalVolume*   poslv             = poscurrentPV->GetLogicalVolume();
//  G4Material*       posmat            = poslv->GetMaterial();
//  std::string posnameVolume;
//  posnameVolume.assign(posname,0,20);

#ifdef ddebug
     std::cout << "============posStep: information:============" << std::endl;
     std::cout << " posposition = "    << posposition
          << " poslocalpoint = "  << poslocalpoint
          << " posvolume = "      << posname
       //          << " posnameVolume = "  << posnameVolume 
          << std::endl;

     std::cout << " ==========================================" << std::endl;
#endif


*/

  //      //
  //      //

  // 24.01.2006:
  // tr     :   id    parentID   trackstatus   tracklength   curstepnumber  entot  vert_pos
  // st     :   stepl EnerDeposit
  // prestep:   preposition   prevolume = SBSTm SIDETL SIDETR       name= SISTATION  copy= 1,2,3    name= SIPLANE  copy= 1..5..10

  // gen_track:
  //  id=1    parentID=1   trackstatus=0,2   tracklength(accumulated)  curstepnumber   entot  vert_pos
  if (id == 1) {
    // on 1-st step:
    if (curstepnumber == 1) {
      entot0 = entot;
      //UserNtuples->fillg519(entot0,1.);
    }

    // on every step:

    // for Copper:
    if (prename == "SBST") {
      SumStepc += stepl;
      // =========
    }
    // for ststeel:
    //	 if(prename == "SBSTs") {
    if (prename == "SBSTs") {
      SumStepl += stepl;
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

        //	   double numbcont = 10.*(copyno1[2]-1)+copyno1[3];

        // =========
        //	   double   xx    = preposition.x();
        //	   double   yy    = preposition.y();
        //	   double   zz    = preposition.z();
        // =========
        //UserNtuples->fillg580(theta,1.);
        //UserNtuples->fillg07(phigrad,1.);
        //	     double xPrimAtZhit = vert_pos.x() + (zz-vert_pos.z())*tan(theta)*cos(phi);
        //	     double yPrimAtZhit = vert_pos.y() + (zz-vert_pos.z())*tan(theta)*sin(phi);
        // =========
        //	   double  dx = xPrimAtZhit - xx;
        //	   double  dy = yPrimAtZhit - yy;
        // =========

        //	   //UserNtuples->fillp212(numbcont,dx,1.);
        //	   //UserNtuples->fillp213(numbcont,dy,1.);
        // =========

        // last step of current SD Volume:
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

    // deposition of energy on steps along primary track
    //UserNtuples->fillg500(EnerDeposit,1.);
    // collect sum deposited energy on all steps along primary track
    SumEnerDeposit += EnerDeposit;
    // position of step for primary track:
    //UserNtuples->fillg501(preposition.x(),1.);
    //UserNtuples->fillg502(preposition.y(),1.);
    //UserNtuples->fillg503(preposition.z(),1.);
    //UserNtuples->fillg504(preposition.x(),EnerDeposit);
    //UserNtuples->fillg505(preposition.y(),EnerDeposit);
    //UserNtuples->fillg506(preposition.z(),EnerDeposit);
    // 2D step coordinates weighted by energy deposited on step
    //      //UserNtuples->fillp201(preposition.x(),preposition.y(),EnerDeposit);
    //      //UserNtuples->fillp202(preposition.x(),preposition.z(),EnerDeposit);
    //      //UserNtuples->fillp203(preposition.y(),preposition.z(),EnerDeposit);
    //UserNtuples->filld204(preposition.x(),preposition.y(),EnerDeposit);
    //UserNtuples->filld205(preposition.x(),preposition.z(),EnerDeposit);
    //UserNtuples->filld206(preposition.y(),preposition.z(),EnerDeposit);
    //UserNtuples->filld223(preposition.y(),preposition.z(),EnerDeposit);
    // last step of primary track
    if (trackstatus == 2) {
      // primary track length
      //      //UserNtuples->fillg508(tracklength,1.);
      tracklength0 = tracklength;
      // how many steps primary track consist
      //UserNtuples->fillg509(curstepnumber,1.);
      // tot. energy of primary track at the end of trajectory(before disappeare)
      //UserNtuples->fillg510((entot0-entot),1.);
      //UserNtuples->fillg520((entot0-entot),1.);
    }
  }
  // end of primary track !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (parentID == 1 && curstepnumber == 1) {
    // particles deposit their energy along primary track
    numofpart += 1;
    // energy of radiated particle
    //UserNtuples->fillg511(entot,1.);
    // coordinates  of radiated particle
    //UserNtuples->fillg512(vert_pos.x(),1.);
    //UserNtuples->fillg513(vert_pos.y(),1.);
    //UserNtuples->fillg514(vert_pos.z(),1.);
    //UserNtuples->fillg515(vert_pos.x(),entot);
    //UserNtuples->fillg516(vert_pos.y(),entot);
    //UserNtuples->fillg517(vert_pos.z(),entot);

    //UserNtuples->filld214(vert_pos.x(),vert_pos.y(),1.);
    //UserNtuples->filld215(vert_pos.x(),vert_pos.z(),1.);
    //UserNtuples->filld216(vert_pos.y(),vert_pos.z(),1.);
    //UserNtuples->filld219(vert_pos.y(),vert_pos.z(),1.);
    //UserNtuples->filld224(vert_pos.y(),vert_pos.z(),1.);
    if (prename == "SBST") {
      //UserNtuples->filld225(vert_pos.y(),vert_pos.z(),1.);
    }
    if (prename == "SBSTs") {
      //UserNtuples->filld226(vert_pos.y(),vert_pos.z(),1.);
    }
  }

  // ==========================================================================
}
// ==========================================================================
// ==========================================================================
int FP420Test::detLevels(const G4VTouchable* touch) const {
  //Return number of levels
  if (touch)
    return ((touch->GetHistoryDepth()) + 1);
  else
    return 0;
}
// ==========================================================================

G4String FP420Test::detName(const G4VTouchable* touch, int level, int currentlevel) const {
  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel;
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void FP420Test::detectorLevel(const G4VTouchable* touch, int& level, int* copyno, G4String* name) const {
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
void FP420Test::update(const EndOfEvent* evt) {
  // ==========================================================================

  if (verbosity > 1) {
    iev = (*evt)()->GetEventID();
    std::cout << "FP420Test:update EndOfEvent = " << iev << std::endl;
  }
  // Fill-in ntuple
  fp420eventarray[ntfp420_evt] = (float)whichevent;

  //
  int trackID = 0;
  G4PrimaryParticle* thePrim = nullptr;

  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  if (nvertex != 1)
    std::cout << "FP420Test: My warning: NumberOfPrimaryVertex != 1  -->  = " << nvertex << std::endl;

  for (int i = 0; i < nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == nullptr)
      std::cout << "FP420Test  End Of Event ERR: pointer to vertex = 0" << std::endl;
    G4int npart = avertex->GetNumberOfParticle();
    if (npart != 1)
      std::cout << "FP420Test: My warning: NumberOfPrimaryPart != 1  -->  = " << npart << std::endl;
    if (npart == 0)
      std::cout << "FP420Test End Of Event ERR: no NumberOfParticle" << std::endl;

    // find just primary track:                                                             track pointer: thePrim
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
      TheHistManager->GetHisto("VtxX")->Fill(vx);
      TheHistManager->GetHisto("VtxY")->Fill(vy);
      TheHistManager->GetHisto("VtxZ")->Fill(vz);
    }
  }
  // prim.vertex loop end

  //=========================== thePrim != 0 ================================================================================
  if (thePrim != nullptr) {
    //      inline G4ParticleDefinition * GetG4code() const
    //      inline G4PrimaryParticle * GetNext() const
    //      inline G4PrimaryParticle * GetDaughter() const

    // prim.vertex
    //int ivert = 0 ;
    //G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(ivert);
    //G4double vx=0.,vy=0.,vz=0.;
    //vx = avertex->GetX0();
    //vy = avertex->GetY0();
    //vz = avertex->GetZ0();

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
    // works OK:
    //      double  costheta =mom.z()/sqrt(mom.x()*mom.x()+mom.y()*mom.y()+mom.z()*mom.z());
    //      double theta = acos(min(max(costheta,double(-1.)),double(1.)));

    //UserNtuples->fillh04(eta);
    //UserNtuples->fillh05(phigrad);
    //UserNtuples->fillh06(th);

    //UserNtuples->fillg13(lastpo.x(),1.);
    //UserNtuples->fillg14(lastpo.y(),1.);
    //UserNtuples->fillg15(lastpo.z(),1.);

    TheHistManager->GetHisto("PrimaryEta")->Fill(eta);
    TheHistManager->GetHisto("PrimaryPhigrad")->Fill(phigrad);
    TheHistManager->GetHisto("PrimaryTh")->Fill(th * 180. / pi);

    TheHistManager->GetHisto("PrimaryLastpoZ")->Fill(lastpo.z());
    if (lastpo.z() < z4) {
      TheHistManager->GetHisto("PrimaryLastpoX")->Fill(lastpo.x());
      TheHistManager->GetHisto("PrimaryLastpoY")->Fill(lastpo.y());
    }
    if (numofpart > 4) {
      TheHistManager->GetHisto("XLastpoNumofpart")->Fill(lastpo.x());
      TheHistManager->GetHisto("YLastpoNumofpart")->Fill(lastpo.y());
    }

    // ==========================================================================

    // hit map for FP420
    // ==================================

    map<int, float, less<int> > themap;
    map<int, float, less<int> > themap1;

    map<int, float, less<int> > themapxy;
    map<int, float, less<int> > themapz;
    // access to the G4 hit collections:  -----> this work OK:

    //  edm::LogInfo("FP420Test") << "1";
    G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
    //  edm::LogInfo("FP420Test") << "2";
    if (verbosity > 0) {
      std::cout << "FP420Test:  accessed all HC" << std::endl;
      ;
    }
    int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("FP420SI");
    // edm::LogInfo("FP420Test") << "3";
    // std::cout << " CAFIid = " << CAFIid << std::endl;;

    FP420G4HitCollection* theCAFI = (FP420G4HitCollection*)allHC->GetHC(CAFIid);
    //  CaloG4HitCollection* theCAFI = (CaloG4HitCollection*) allHC->GetHC(CAFIid);
    if (verbosity > 0) {
      //std::cout << "FP420Test: theCAFI = " << theCAFI << std::endl;
      std::cout << "FP420Test: theCAFI->entries = " << theCAFI->entries() << std::endl;
    }
    // edm::LogInfo("FP420Test") << "theCAFI->entries="<< theCAFI->entries();
    TheHistManager->GetHisto("NumberOfHits")->Fill(theCAFI->entries());

    // access to the G4 hit collections ----> variant 2: give 0 hits
    //  FP420G4HitCollection *   theCAFI;
    //  theCAFI = new FP420G4HitCollection();
    // ==========================================================================
    //   Silicon Hit collection start
    //0) if particle goes into flat beam pipe below detector:
    int varia;  // = 0 -all; =1 - MI; =2 - noMI
    //                      Select MI or noMI over all 3 stations
    // 1)MI:
    //     if particle goes through window into detector:
    // lastpoint of track in lateral dir. outside the detector and inside in z
    // lastpoint of track in lateral dir. outside the detector and inside in z
    // for all except zzzmarta.xml
    //    if(  lastpo.z()<z4  ||  abs(lastpo.x())> 5. || lastpo.y()< 10.2 || lastpo.y()>30.2   ) {
    // for zzzmarta.xml
    //   if(  lastpo.z()<z4  ||  abs(lastpo.x())> 10. || lastpo.y()< 3.2 || lastpo.y()>43.2   ) {
    if (lastpo.z() < z4) {
      //  if(  lastpo.z()<z4 && lastpo.perp()< 100. ) {
      //  if(lastpo.z()<z4  || lastpo.perp()> 45.) {
      //UserNtuples->fillg66(theCAFI->entries(),1.);
      varia = 1;
    } else {
      // 2)   no MI start in detector:
      //UserNtuples->fillg31(numofpart,1.);
      //UserNtuples->fillg67(theCAFI->entries(),1.);
      varia = 2;
    }  // no MI end:
    int nhits = theCAFI->entries();
    for (int j = 0; j < nhits; j++) {
      FP420G4Hit* aHit = (*theCAFI)[j];
      G4ThreeVector hitPoint = aHit->getEntry();
      double zz = hitPoint.z();
      TheHistManager->GetHisto("zHits")->Fill(zz);
      if (tracklength0 > 8300.)
        TheHistManager->GetHisto("zHitsTrLoLe")->Fill(zz);
    }
    // varia = 0;
    //     if( varia == 0) {
    if (varia == 2) {
      // .............
      // number of hits < 50
      //    if(theCAFI->entries() <50) {
      //    if(theCAFI->entries() > 0) {
      //    if(theCAFI->entries() > -1) {
      // .............
      int nhit11 = 0, nhit12 = 0, nhit13 = 0;
      double totallosenergy = 0.;
      for (int j = 0; j < nhits; j++) {
        FP420G4Hit* aHit = (*theCAFI)[j];

        G4ThreeVector hitEntryLocalPoint = aHit->getEntryLocalP();
        G4ThreeVector hitExitLocalPoint = aHit->getExitLocalP();
        G4ThreeVector hitPoint = aHit->getEntry();
        //    double  elmenergy =  aHit->getEM();
        //    double  hadrenergy =  aHit->getHadr();
        //    double incidentEnergyHit  = aHit->getIncidentEnergy();
        int trackIDhit = aHit->getTrackID();
        unsigned int unitID = aHit->getUnitID();
        //    double   timeslice = aHit->getTimeSlice();
        //    int     timesliceID = aHit->getTimeSliceID();
        //    double  depenergy = aHit->getEnergyDeposit();
        //    float   pabs = aHit->getPabs();
        //    float   tof = aHit->getTof();
        double losenergy = aHit->getEnergyLoss();
        //    int   particletype = aHit->getParticleType();
        //    float thetaEntry = aHit->getThetaAtEntry();
        //    float phiEntry = aHit->getPhiAtEntry();
        //    float xEntry = aHit->getX();
        //    float yEntry = aHit->getY();
        //    float zEntry = aHit->getZ();
        //    int  parentID = aHit->getParentId();
        //    float vxget = aHit->getVx();
        //    float vyget = aHit->getVy();
        //    float vzget = aHit->getVz();

        //    double th_hit    = hitPoint.theta();
        //    double eta_hit = -log(tan(th_hit/2));
        //    double phi_hit   = hitPoint.phi();
        //    if (phi_hit < 0.) phi_hit += twopi;
        //    double phigrad_hit = phi_hit*180./pi;
        //UserNtuples->fillg60(eta_hit,losenergy);
        //UserNtuples->fillg61(eta_hit,1.);
        //UserNtuples->fillg62(phigrad_hit,losenergy);
        //UserNtuples->fillg63(phigrad_hit,1.);

        //    double   xx    = hitPoint.x();
        //    double   yy    = hitPoint.y();
        double zz = hitPoint.z();

        TheHistManager->GetHisto("zHitsnoMI")->Fill(zz);

        if (verbosity > 2) {
          std::cout << "FP420Test:zHits = " << zz << std::endl;
        }
        //	 double   rr    = hitPoint.perp();
        /*
      if(aHit->getTrackID() == 1) {
	  emu += aHit->getEnergyDeposit();} else {
	  erest += aHit->getEnergyDeposit();}
    */

        //collect lost in Si en.of hits in every plane and put it into themap[]
        //UserNtuples->fillg30(losenergy,1.);
        themap[unitID] += losenergy;
        totallosenergy += losenergy;

        int det, zside, sector, zmodule;
        //    CaloNumberingPacker::unpackCastorIndex(unitID, det, zside, sector, zmodule);
        FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
        int justlayer = FP420NumberingScheme::unpackLayerIndex(rn00, zside);  // 1,2
        if (justlayer < 1 || justlayer > 2) {
          std::cout << "FP420Test:WRONG  justlayer= " << justlayer << std::endl;
        }
        // zside=1,2 ; zmodule=1,10 ; sector=1,3
        //UserNtuples->fillg44(float(sector),1.);
        //UserNtuples->fillg45(float(zmodule),1.);
        //UserNtuples->fillg46(float(zside),1.);
        //      int sScale = 20;
        // intindex is a continues numbering of FP420
        //int zScale = 2; unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside; //intindex=1-30:X,Y,X,Y,X,Y...
        // int zScale = 10;   unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule; //intindex=1-30:XXXXXXXXXX,YYYYYYYYYY,...
        //UserNtuples->fillg40(float(intindex),1.);
        //UserNtuples->fillg48(float(intindex),losenergy);
        //
        //=======================================
        //   G4ThreeVector middle = (hitExitLocalPoint+hitEntryLocalPoint)/2.;
        G4ThreeVector middle = (hitExitLocalPoint - hitEntryLocalPoint) / 2.;
        themapz[unitID] = hitPoint.z() + fabs(middle.z());
        if (verbosity > 2) {
          std::cout << "1111111111111111111111111111111111111111111111111111111111111111111111111 " << std::endl;
          std::cout << "FP420Test: det, zside, sector, zmodule = " << det << zside << sector << zmodule << std::endl;
          std::cout << "FP420Test: justlayer = " << justlayer << std::endl;
          std::cout << "FP420Test: hitExitLocalPoint = " << hitExitLocalPoint << std::endl;
          std::cout << "FP420Test: hitEntryLocalPoint = " << hitEntryLocalPoint << std::endl;
          std::cout << "FP420Test:  middle= " << middle << std::endl;
          std::cout << "FP420Test:  hitPoint.z()-419000.= " << hitPoint.z() - 419000. << std::endl;

          std::cout << "FP420Test:zHits-419000. = " << themapz[unitID] - 419000. << std::endl;
        }
        //=======================================
        // Y
        if (zside == 1) {
          //UserNtuples->fillg24(losenergy,1.);
          if (losenergy > 0.00003) {
            themap1[unitID] += 1.;
          }
        }
        //X
        if (zside == 2) {
          //UserNtuples->fillg25(losenergy,1.);
          if (losenergy > 0.00005) {
            themap1[unitID] += 1.;
          }
        }
        //	   }
        //
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
        //UserNtuples->fillg10(xx,1.);
        //UserNtuples->fillg12(zz,1.);
        //UserNtuples->fillg32(rr,1.);

        // =========
        //    double xPrimAtZhit = vx + (zz-vz)*tan(th)*cos(phi);
        //    double yPrimAtZhit = vy + (zz-vz)*tan(th)*sin(phi);

        //       double  dx = xPrimAtZhit - xx;
        //       double  dy = yPrimAtZhit - yy;

        //                      Select SD hits
        //    if(rr<120.) {
        //                      Select MI or noMI over all 3 stations
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
            //UserNtuples->fillg21(SumEnerDeposit,1.);
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
            //UserNtuples->fillg23(SumEnerDeposit,1.);
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
            //UserNtuples->fillg68(dx,1.);
            //UserNtuples->fillg69(dy,1.);
            //UserNtuples->filld210(xx,yy,1.);
            //UserNtuples->fillg22(numofpart,1.);
            //UserNtuples->fillg23(SumEnerDeposit,1.);
            if (zside == 1) {
              //UserNtuples->fillg28(losenergy,1.);
            }
            if (zside == 2) {
              //UserNtuples->fillg78(losenergy,1.);
            }
          }
        }  // MIonly or noMIonly ENDED
           //    }

        //     !!!!!!!!!!!!!

      }  // for loop on all hits ENDED  ENDED  ENDED  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      //     !!!!!!!!!!!!!

      //======================================================================================================SUMHIT
      //UserNtuples->fillg29(totallosenergy,1.);
      //UserNtuples->fillg36(nhit11,1.);
      //UserNtuples->fillg37(nhit12,1.);
      //UserNtuples->fillg38(nhit13,1.);
      //======================================================================================================SUMHIT
      //   int rn00=3;//test only with 2 sensors in superlayer, not 4
      //	  int rn00=rn0;//always
      if (verbosity > 2) {
        std::cout << "22222222222222222222222222222222222222222222222222222222222222222222222222 " << std::endl;
      }
      int det = 1;
      int allplacesforsensors = 7;
      for (int sector = 1; sector < sn0; sector++) {
        for (int zmodule = 1; zmodule < pn0; zmodule++) {
          for (int zsideinorder = 1; zsideinorder < allplacesforsensors; zsideinorder++) {
            int zside = FP420NumberingScheme::realzside(rn00, zsideinorder);  //1,3,5,2,4,6
            if (verbosity > 2) {
              std::cout << "FP420Test:  sector= " << sector << " zmodule= " << zmodule
                        << " zsideinorder= " << zsideinorder << " zside= " << zside << std::endl;
            }
            if (zside != 0) {
              int justlayer = FP420NumberingScheme::unpackLayerIndex(rn00, zside);  // 1,2
              if (justlayer < 1 || justlayer > 2) {
                std::cout << "FP420Test:WRONG  justlayer= " << justlayer << std::endl;
              }
              int copyinlayer = FP420NumberingScheme::unpackCopyIndex(rn00, zside);  // 1,2,3
              if (copyinlayer < 1 || copyinlayer > 3) {
                std::cout << "FP420Test:WRONG  copyinlayer= " << copyinlayer << std::endl;
              }
              int orientation = FP420NumberingScheme::unpackOrientation(rn00, zside);  // Front: = 1; Back: = 2
              if (orientation < 1 || orientation > 2) {
                std::cout << "FP420Test:WRONG  orientation= " << orientation << std::endl;
              }

              // iu is a continues numbering of planes(!)  over two arm FP420 set up
              int detfixed =
                  1;  // use this treatment for each set up arm, hence no sense to do it defferently for +FP420 and -FP420;
              //                                                                    and  ...[ii] massives have prepared in such a way
              unsigned int ii =
                  FP420NumberingScheme::packMYIndex(rn00, pn0, sn0, detfixed, justlayer, sector, zmodule) - 1;
              // ii = 0-19   --> 20 items
              if (verbosity > 2) {
                std::cout << "FP420Test:  justlayer = " << justlayer << " copyinlayer = " << copyinlayer
                          << " orientation = " << orientation << " ii= " << ii << std::endl;
              }
              double zdiststat = 0.;
              if (sn0 < 4) {
                if (sector == 2)
                  zdiststat = zD3;
              } else {
                if (sector == 2)
                  zdiststat = zD2;
                if (sector == 3)
                  zdiststat = zD3;
              }
              double kplane = -(pn0 - 1) / 2 - 0.5 + (zmodule - 1);  //-3.5 +0...5 = -3.5,-2.5,-1.5,+2.5,+1.5
              double zcurrent = zinibeg + z420 + (ZSiStep - ZSiPlane) / 2 + kplane * ZSiStep + zdiststat;
              //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;
              if (verbosity > 2) {
                std::cout << "FP420Test:  Leftzcurrent-419000. = " << zcurrent - 419000. << std::endl;
                std::cout << "FP420Test:  ZGapLDet = " << ZGapLDet << std::endl;
              }
              if (justlayer == 1) {
                if (orientation == 1)
                  zcurrent += (ZGapLDet + ZSiDet / 2);
                if (orientation == 2)
                  zcurrent += zBlade - (ZGapLDet + ZSiDet / 2);
              }
              if (justlayer == 2) {
                if (orientation == 1)
                  zcurrent += (ZGapLDet + ZSiDet / 2) + zBlade + gapBlade;
                if (orientation == 2)
                  zcurrent += 2 * zBlade + gapBlade - (ZGapLDet + ZSiDet / 2);
              }
              //   .
              //
              if (det == 2)
                zcurrent = -zcurrent;
              //
              if (verbosity > 2) {
                std::cout << "FP420Test:  zcurrent-419000. = " << zcurrent - 419000. << std::endl;
              }
              //================================== end of for loops in continuius number iu:
            }  //if(zside!=0
          }    // for superlayer
        }      // for zmodule
      }        // for sector

      if (verbosity > 2) {
        std::cout << "----------------------------------------------------------------------------- " << std::endl;
      }

      //======================================================================================================CHECK
      if (totallosenergy == 0.0) {
        std::cout << "FP420Test:     number of hits = " << theCAFI->entries() << std::endl;
        for (int j = 0; j < nhits; j++) {
          FP420G4Hit* aHit = (*theCAFI)[j];
          double losenergy = aHit->getEnergyLoss();
          std::cout << " j hits = " << j << "losenergy = " << losenergy << std::endl;
        }
      }
      //======================================================================================================CHECK

      //====================================================================================================== HIT  START

      //   FIBRE Hit collected analysis
      double totalEnergy = 0.;
      int nhitsX = 0, nhitsY = 0, nsumhit = 0;
      for (int sector = 1; sector < 4; sector++) {
        int nhitsecX = 0, nhitsecY = 0;
        for (int zmodule = 1; zmodule < 11; zmodule++) {
          for (int zside = 1; zside < 3; zside++) {
            int det = 1;
            //      int nhit = 0;
            //	int sScale = 20;
            int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
            double theTotalEnergy = themap[index];
            //   X planes
            if (zside < 2) {
              //UserNtuples->fillg47(theTotalEnergy,1.);
              if (theTotalEnergy > 0.00003) {
                nhitsX += 1;
                //	    nhitsecX += themap1[index];
                //	    nhit=1;
              }
            }
            //   Y planes
            else {
              //UserNtuples->fillg49(theTotalEnergy,1.);
              if (theTotalEnergy > 0.00005) {
                nhitsY += 1;
                //	    nhitsecY += themap1[index];
                //	    nhit=1;
              }
            }
            // intindex is a continues numbering of FP420
            //        int zScale=2;  unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
            // int zScale=10;       unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
            //UserNtuples->fillg41(float(intindex),theTotalEnergy);
            //UserNtuples->fillg42(float(intindex),1.);
            //UserNtuples->fillp208(float(intindex),float(nhit),1.);
            //UserNtuples->fillp211(float(intindex),float(themap1[index]),1.);
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
      //====================================================================================================== HIT  END

      //====================================================================================================== HIT  ALL
      //UserNtuples->fillg43(totalEnergy,1.);
      //UserNtuples->fillg58(nhitsX,1.);
      //UserNtuples->fillg59(nhitsY,1.);
      //  if( nsumhit !=0 ) { //UserNtuples->fillp212(vy,float(1.),1.);
      if (nsumhit >= 2) {  //UserNtuples->fillp212(vy,float(1.),1.);
      } else {             //UserNtuples->fillp212(vy,float(0.),1.);
      }

      //====================================================================================================== HIT  ALL

      //====================================================================================================== number of hits
      // .............
      //    } // number of hits < 50
      // .............
    }  // MI or no MI or all  - end

  }  // primary end
     //=========================== thePrim != 0  end   ===
  // ==========================================================================
  if (verbosity > 0) {
    std::cout << "FP420Test:  END OF Event " << (*evt)()->GetEventID() << std::endl;
  }
}

// ==========================================================================
