// -*- C++ -*-
//

// system include files
#include <iostream>
#include <iomanip>
#include <cmath>
#include<vector>
//
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// to retreive hits
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "SimG4CMS/Forward/interface/BscTest.h"

//#include "Utilities/GenUtil/interface/CMSexception.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"

// G4 stuff
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "G4TransportationManager.hh"
#include "G4ProcessManager.hh"
//#include "G4EventManager.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <stdio.h>
//#include <gsl/gsl_fit.h>


//================================================================
// Root stuff

// Include the standard header <cassert> to effectively include
// the standard header <assert.h> within the std namespace.
#include <cassert>

//================================================================

//UserVerbosity BscTest::std::cout("BscTest","info","BscTest");
enum ntbsc_elements {
  ntbsc_evt
};

//================================================================
BscTest::BscTest(const edm::ParameterSet &p){
  //constructor
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("BscTest");
  verbosity    = m_Anal.getParameter<int>("Verbosity");
  //verbosity    = 1;

  fDataLabel  = m_Anal.getParameter<std::string>("FDataLabel");
  fOutputFile = m_Anal.getParameter<std::string>("FOutputFile");
  fRecreateFile = m_Anal.getParameter<std::string>("FRecreateFile");
   
  if (verbosity > 0) {
    std::cout<<"============================================================================"<<std::endl;
    std::cout << "BscTestconstructor :: Initialized as observer"<< std::endl;
  }
  // Initialization:

  theBscNumberingScheme = new BscNumberingScheme();
  bsceventntuple = new TNtuple("NTbscevent","NTbscevent","evt");
  whichevent = 0;
  TheHistManager = new BscAnalysisHistManager(fDataLabel);

  if (verbosity > 0) {
    std::cout << "BscTest constructor :: Initialized BscAnalysisHistManager"<< std::endl;
  }
}



BscTest::~BscTest() {
  //  delete UserNtuples;
  delete theBscNumberingScheme;

  TFile bscOutputFile("newntbsc.root","RECREATE");
  std::cout << "Bsc output root file has been created";
  bsceventntuple->Write();
  std::cout << ", written";
  bscOutputFile.Close();
  std::cout << ", closed";
  delete bsceventntuple;
  std::cout << ", and deleted" << std::endl;

  //------->while end

  // Write histograms to file
  TheHistManager->WriteToFile(fOutputFile,fRecreateFile);
  if (verbosity > 0) {
    std::cout << std::endl << "BscTest Destructor  -------->  End of BscTest : " << std::endl;
  }

  std::cout<<"BscTest: End of process"<<std::endl;



}

//================================================================
// Histoes:
//-----------------------------------------------------------------------------

BscAnalysisHistManager::BscAnalysisHistManager(const TString& managername)
{
  // The Constructor

  fTypeTitle=managername;
  fHistArray = new TObjArray();      // Array to store histos
  fHistNamesArray = new TObjArray(); // Array to store histos's names

  BookHistos();

  fHistArray->Compress();            // Removes empty space
  fHistNamesArray->Compress();

  //      StoreWeights();                    // Store the weights

}
//-----------------------------------------------------------------------------

BscAnalysisHistManager::~BscAnalysisHistManager()
{
  // The Destructor

  if(fHistArray){
    fHistArray->Delete();
    delete fHistArray;
  }

  if(fHistNamesArray){
    fHistNamesArray->Delete();
    delete fHistNamesArray;
  }
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::BookHistos()
{
  // at Start: (mm)
  HistInit("TrackPhi", "Primary Phi",   100,   0.,360. );
  HistInit("TrackTheta", "Primary Theta",   100,   0.,180. );
  HistInit("TrackP", "Track XY position Z+ ",  80, -80., 80.,  80, -80., 80. );
  HistInit("TrackM", "Track XY position Z-",   80, -80., 80.,  80, -80., 80. );
  HistInit("DetIDs", "Track DetId - vs +",   16, -0.5, 15.5,16, 15.5, 31.5 );
}

//-----------------------------------------------------------------------------

void BscAnalysisHistManager::WriteToFile(const TString& fOutputFile,const TString& fRecreateFile)
{

  //Write to file = fOutputFile

  std::cout <<"================================================================"<<std::endl;
  std::cout <<" Write this Analysis to File "<<fOutputFile<<std::endl;
  std::cout <<"================================================================"<<std::endl;

  TFile* file = new TFile(fOutputFile, fRecreateFile);

  fHistArray->Write();
  file->Close();
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup)
{
  // Add histograms and histograms names to the array

  char* newtitle = new char[strlen(title)+strlen(fTypeTitle)+5];
  strcpy(newtitle,title);
  strcat(newtitle," (");
  strcat(newtitle,fTypeTitle);
  strcat(newtitle,") ");
  fHistArray->AddLast((new TH1F(name, newtitle, nbinsx, xlow, xup)));
  fHistNamesArray->AddLast(new TObjString(name));

}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)
{
  // Add histograms and histograms names to the array

  char* newtitle = new char[strlen(title)+strlen(fTypeTitle)+5];
  strcpy(newtitle,title);
  strcat(newtitle," (");
  strcat(newtitle,fTypeTitle);
  strcat(newtitle,") ");
  fHistArray->AddLast((new TH2F(name, newtitle, nbinsx, xlow, xup, nbinsy, ylow, yup)));
  fHistNamesArray->AddLast(new TObjString(name));

}
//-----------------------------------------------------------------------------

TH1F* BscAnalysisHistManager::GetHisto(Int_t Number)
{
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast()  && fHistArray->At(Number) != (TObject*)0){

    return (TH1F*)(fHistArray->At(Number));

  }else{

    std::cout << "!!!!!!!!!!!!!!!!!!GetHisto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    return (TH1F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH2F* BscAnalysisHistManager::GetHisto2(Int_t Number)
{
  // Get a histogram from the array with index = Number

  if (Number <= fHistArray->GetLast()  && fHistArray->At(Number) != (TObject*)0){

    return (TH2F*)(fHistArray->At(Number));

  }else{

    std::cout << "!!!!!!!!!!!!!!!!GetHisto2!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " WARNING ERROR - HIST ID INCORRECT (TOO HIGH) - " << Number << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

    return (TH2F*)(fHistArray->At(0));
  }
}
//-----------------------------------------------------------------------------

TH1F* BscAnalysisHistManager::GetHisto(const TObjString& histname)
{
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return GetHisto(index);
}
//-----------------------------------------------------------------------------

TH2F* BscAnalysisHistManager::GetHisto2(const TObjString& histname)
{
  // Get a histogram from the array with name = histname

  Int_t index = fHistNamesArray->IndexOf(&histname);
  return GetHisto2(index);
}
//-----------------------------------------------------------------------------

void BscAnalysisHistManager::StoreWeights()
{
  // Add structure to each histogram to store the weights

  for(int i = 0; i < fHistArray->GetEntries(); i++){
    ((TH1F*)(fHistArray->At(i)))->Sumw2();
  }
}


//==================================================================== per JOB
void BscTest::update(const BeginOfJob * job) {
  //job
  std::cout<<"BscTest:beggining of job"<<std::endl;;
}


//==================================================================== per RUN
void BscTest::update(const BeginOfRun * run) {
  //run

  std::cout << std::endl << "BscTest:: Begining of Run"<< std::endl; 
}


void BscTest::update(const EndOfRun * run) {;}



//=================================================================== per EVENT
void BscTest::update(const BeginOfEvent * evt) {
  iev = (*evt)()->GetEventID();
  if (verbosity > 0) {
    std::cout <<"BscTest:update Event number = " << iev << std::endl;
  }
  whichevent++;
}

//=================================================================== per Track
void BscTest::update(const BeginOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    std::cout <<"BscTest:update BeginOfTrack number = " << itrk << std::endl;
  }
  if(itrk == 1) {
    SumEnerDeposit = 0.;
    numofpart = 0;
    SumStepl = 0.;
    SumStepc = 0.;
    tracklength0 = 0.;
  }
}



//=================================================================== per EndOfTrack
void BscTest::update(const EndOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  if (verbosity > 1) {
    std::cout <<"BscTest:update EndOfTrack number = " << itrk << std::endl;
  }
  if(itrk == 1) {
    G4double tracklength  = (*trk)()->GetTrackLength();    // Accumulated track length

    TheHistManager->GetHisto("SumEDep")->Fill(SumEnerDeposit);
    TheHistManager->GetHisto("TrackL")->Fill(tracklength);

    // direction !!!
    G4ThreeVector   vert_mom  = (*trk)()->GetVertexMomentumDirection();
    G4ThreeVector   vert_pos  = (*trk)()->GetVertexPosition(); // vertex ,where this track was created
  
    //    float eta = 0.5 * log( (1.+vert_mom.z()) / (1.-vert_mom.z()) );
    /*
    float phi = atan2(vert_mom.y(),vert_mom.x());
    if (phi < 0.) phi += twopi;
    if(tracklength < z4) {

    }
    */
    // last step information
    const G4Step* aStep = (*trk)()->GetStep();
    G4StepPoint*      preStepPoint = aStep->GetPreStepPoint(); 
    lastpo   = preStepPoint->GetPosition();	

    // Analysis:

  }

}

// ====================================================

//=================================================================== each STEP
void BscTest::update(const G4Step * aStep) {
  // ==========================================================================
  
  if (verbosity > 2) {
    G4int stepnumber  = aStep->GetTrack()->GetCurrentStepNumber();
    std::cout <<"BscTest:update Step number = " << stepnumber << std::endl;
  }
  // track on aStep:                                                                                         !
  G4Track*     theTrack     = aStep->GetTrack();   
  TrackInformation* trkInfo = dynamic_cast<TrackInformation*> (theTrack->GetUserInformation());
  if (trkInfo == 0) {
    std::cout << "BscTest on aStep: No trk info !!!! abort " << std::endl;
    //     throw Genexception("BscTest:BscTest on aStep: cannot get trkInfo");
  } 
  G4int         id             = theTrack->GetTrackID();
  G4String       particleType   = theTrack->GetDefinition()->GetParticleName();   //   !!!
  G4int         parentID       = theTrack->GetParentID();   //   !!!
  G4TrackStatus   trackstatus    = theTrack->GetTrackStatus();   //   !!!
  G4double       tracklength    = theTrack->GetTrackLength();    // Accumulated track length
  G4ThreeVector   trackmom       = theTrack->GetMomentum();
  G4double       entot          = theTrack->GetTotalEnergy();   //   !!! deposited on step
  G4int         curstepnumber  = theTrack->GetCurrentStepNumber();
  G4ThreeVector   vert_pos       = theTrack->GetVertexPosition(); // vertex ,where this track was created
  G4ThreeVector   vert_mom       = theTrack->GetVertexMomentumDirection();
  G4double        stepl         = aStep->GetStepLength();
  G4double        EnerDeposit   = aStep->GetTotalEnergyDeposit();
  G4StepPoint*      preStepPoint = aStep->GetPreStepPoint(); 
  G4ThreeVector     preposition   = preStepPoint->GetPosition();	
  G4ThreeVector     prelocalpoint = theTrack->GetTouchable()->GetHistory()->
    GetTopTransform().TransformPoint(preposition);
  G4VPhysicalVolume* currentPV     = preStepPoint->GetPhysicalVolume();
  G4String         prename       = currentPV->GetName();

  const G4VTouchable*  pre_touch    = preStepPoint->GetTouchable();
  int          pre_levels   = detLevels(pre_touch);
  G4String name1[20]; int copyno1[20];
  if (pre_levels > 0) {
    detectorLevel(pre_touch, pre_levels, copyno1, name1);
  }

  if ( id == 1 ) {

    // on 1-st step:
    if (curstepnumber == 1 ) {
      entot0 = entot;
      //UserNtuples->fillg519(entot0,1.);

    }

    // on every step:

    // for Copper:
    if(prename == "SBST" ) {
      SumStepc += stepl;
      // =========
    }
    // for ststeel:
    //	 if(prename == "SBSTs") {
    if(prename == "SBSTs" ) {
      SumStepl += stepl;
      // =========
    }
    // =========
    // =========

    // exclude last track point if it is in SD (MI was started their)
    if (trackstatus != 2 ) {
      // for SD:   Si Det.:   SISTATION:SIPLANE:(SIDETL+BOUNDDET        +SIDETR + CERAMDET)
      if(prename == "SIDETL" || prename == "SIDETR" ) {
	if(prename == "SIDETL") {
	  //UserNtuples->fillg569(EnerDeposit,1.);
	}
	if(prename == "SIDETR") {
	  //UserNtuples->fillg570(EnerDeposit,1.);
	}

	G4String posname = aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName();
	if((prename == "SIDETL" && posname != "SIDETL") || (prename == "SIDETR" && posname != "SIDETR")) {
	  if(name1[2] == "SISTATION" ) {
	    //UserNtuples->fillg539(copyno1[2],1.);
	  }
	  if(name1[3] == "SIPLANE" ) {
	    //UserNtuples->fillg540(copyno1[3],1.);
	  }

	  if(prename == "SIDETL") {
	    //UserNtuples->fillg541(EnerDeposit,1.);
	    //UserNtuples->fillg561(numbcont,1.);
	    if(copyno1[2]<2) {
	      //UserNtuples->fillg571(dx,1.);
	    }
	    else if(copyno1[2]<3) {
	      //UserNtuples->fillg563(dx,1.);
	      if(copyno1[3]<2) {
	      }
	      else if(copyno1[3]<3) {
		//UserNtuples->fillg572(dx,1.);
	      }
	      else if(copyno1[3]<4) {
		//UserNtuples->fillg573(dx,1.);
	      }
	      else if(copyno1[3]<5) {
		//UserNtuples->fillg574(dx,1.);
	      }
	      else if(copyno1[3]<6) {
		//UserNtuples->fillg575(dx,1.);
	      }
	      else if(copyno1[3]<7) {
		//UserNtuples->fillg576(dx,1.);
	      }
	      else if(copyno1[3]<8) {
		//UserNtuples->fillg577(dx,1.);
	      }
	      else if(copyno1[3]<9) {
		//UserNtuples->fillg578(dx,1.);
	      }
	      else if(copyno1[3]<10) {
		//UserNtuples->fillg579(dx,1.);
	      }
	    }
	    else if(copyno1[2]<4) {
	      //UserNtuples->fillg565(dx,1.);
	    }
	    else if(copyno1[2]<5) {
	      //UserNtuples->fillg567(dx,1.);
	    }
	  }
	  if(prename == "SIDETR") {
	    //UserNtuples->fillg542(EnerDeposit,1.);
	    //UserNtuples->fillg562(numbcont,1.);
	    if(copyno1[2]<2) {
	      //UserNtuples->fillg581(dy,1.);
	    }
	    else if(copyno1[2]<3) {
	      //UserNtuples->fillg564(dy,1.);
	      if(copyno1[3]<2) {
	      }
	      else if(copyno1[3]<3) {
		//UserNtuples->fillg582(dy,1.);
	      }
	      else if(copyno1[3]<4) {
		//UserNtuples->fillg583(dy,1.);
	      }
	      else if(copyno1[3]<5) {
		//UserNtuples->fillg584(dy,1.);
	      }
	      else if(copyno1[3]<6) {
		//UserNtuples->fillg585(dy,1.);
	      }
	      else if(copyno1[3]<7) {
		//UserNtuples->fillg586(dy,1.);
	      }
	      else if(copyno1[3]<8) {
		//UserNtuples->fillg587(dy,1.);
	      }
	      else if(copyno1[3]<9) {
		//UserNtuples->fillg588(dy,1.);
	      }
	      else if(copyno1[3]<10) {
		//UserNtuples->fillg589(dy,1.);
	      }
	    }
	    else if(copyno1[2]<4) {
	      //UserNtuples->fillg566(dy,1.);
	    }
	    else if(copyno1[2]<5) {
	      //UserNtuples->fillg568(dy,1.);
	    }
	  }

	}
      }
      // end of prenames SIDETL // SIDETR
    }
    // end of trackstatus != 2

    SumEnerDeposit += EnerDeposit;
    if (trackstatus == 2 ) {
      // primary track length 
      //      //UserNtuples->fillg508(tracklength,1.);
      tracklength0 = tracklength;
    }
  }
  // end of primary track !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  if (parentID == 1 && curstepnumber == 1) {
    // particles deposit their energy along primary track
    numofpart += 1;
    if(prename == "SBST" ) {
      //UserNtuples->filld225(vert_pos.y(),vert_pos.z(),1.);
    }
    if(prename == "SBSTs" ) {
      //UserNtuples->filld226(vert_pos.y(),vert_pos.z(),1.);
    }
  }

}
// ==========================================================================
// ==========================================================================
int BscTest::detLevels(const G4VTouchable* touch) const {

  //Return number of levels
  if (touch) 
    return ((touch->GetHistoryDepth())+1);
  else
    return 0;
}
// ==========================================================================

G4String BscTest::detName(const G4VTouchable* touch, int level,
			  int currentlevel) const {

  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel; 
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void BscTest::detectorLevel(const G4VTouchable* touch, int& level,
			    int* copyno, G4String* name) const {

  //Get name and copy numbers
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      G4VPhysicalVolume* pv = touch->GetVolume(i);
      if (pv != 0) 
        name[ii] = pv->GetName();
      else
        name[ii] = "Unknown";
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
// ==========================================================================

//===================================================================   End Of Event
void BscTest::update(const EndOfEvent * evt) {
  // ==========================================================================
  
  if (verbosity > 1) {
    iev = (*evt)()->GetEventID();
    std::cout <<"BscTest:update EndOfEvent = " << iev << std::endl;
  }
  // Fill-in ntuple
  bsceventarray[ntbsc_evt] = (float)whichevent;

  //
  int trackID = 0;
  G4PrimaryParticle* thePrim=0;


  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  if (nvertex !=1)
    std::cout << "BscTest: My warning: NumberOfPrimaryVertex != 1  -->  = " << nvertex <<  std::endl;

  for (int i = 0 ; i<nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == 0)
      std::cout << "BscTest  End Of Event ERR: pointer to vertex = 0"
		<< std::endl;
    G4int npart = avertex->GetNumberOfParticle();
    if (npart !=1)
      std::cout << "BscTest: My warning: NumberOfPrimaryPart != 1  -->  = " << npart <<  std::endl;
    if (npart ==0)
      std::cout << "BscTest End Of Event ERR: no NumberOfParticle" << std::endl;

    if (thePrim==0) thePrim=avertex->GetPrimary(trackID);

    if (thePrim!=0) {
      // primary vertex:
      G4double vx=0.,vy=0.,vz=0.;
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
  if (thePrim != 0) {
    //      inline G4ParticleDefinition * GetG4code() const
    //      inline G4PrimaryParticle * GetNext() const
    //      inline G4PrimaryParticle * GetDaughter() const
    /*
    // prim.vertex
    int ivert = 0 ;
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(ivert);
    G4double vx=0.,vy=0.,vz=0.;
    vx = avertex->GetX0();
    vy = avertex->GetY0();
    vz = avertex->GetZ0();
    */
    //
    // number of secondary particles deposited their energy along primary track
    //UserNtuples->fillg518(numofpart,1.);
    if(lastpo.z()<z4 && lastpo.perp()< 100.) {
      //UserNtuples->fillg536(numofpart,1.);
    }
    //

    // direction !!!
    G4ThreeVector   mom  = thePrim->GetMomentum();
  
    double phi = atan2(mom.y(),mom.x());
    if (phi < 0.) phi += twopi;
    double phigrad = phi*180./pi;

    double th     = mom.theta();
    double eta = -log(tan(th/2));
    TheHistManager->GetHisto("PrimaryEta")->Fill(eta);
    TheHistManager->GetHisto("PrimaryPhigrad")->Fill(phigrad);
    TheHistManager->GetHisto("PrimaryTh")->Fill(th*180./pi);

    TheHistManager->GetHisto("PrimaryLastpoZ")->Fill(lastpo.z());
    if(lastpo.z() <  z4  ) {
      TheHistManager->GetHisto("PrimaryLastpoX")->Fill(lastpo.x());
      TheHistManager->GetHisto("PrimaryLastpoY")->Fill(lastpo.y());
    }
    if(numofpart >  4  ) {
      TheHistManager->GetHisto("XLastpoNumofpart")->Fill(lastpo.x());
      TheHistManager->GetHisto("YLastpoNumofpart")->Fill(lastpo.y());
    }

    // ==========================================================================

    // hit map for Bsc
    // ==================================

    std::map<int,float,std::less<int> > themap;
    std::map<int,float,std::less<int> > themap1;

    std::map<int,float,std::less<int> > themapxy;
    std::map<int,float,std::less<int> > themapz;
    // access to the G4 hit collections:  -----> this work OK:

    //  edm::LogInfo("BscTest") << "1";
    G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
    //  edm::LogInfo("BscTest") << "2";
    if (verbosity > 0) {
      std::cout << "BscTest:  accessed all HC" << std::endl;;
    }
    int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("BSCHits");

    BscG4HitCollection* theCAFI = (BscG4HitCollection*) allHC->GetHC(CAFIid);
    if (verbosity > 0) {
      std::cout << "BscTest: theCAFI->entries = " << theCAFI->entries() << std::endl;
    }
    int varia ;   // = 0 -all; =1 - MI; =2 - noMI
    //varia = 0;
    if(  lastpo.z()< z4) {
      varia = 1;
    }
    else{
      varia = 2;
    }   // no MI end:
    for (int j=0; j<theCAFI->entries(); j++) {
      BscG4Hit* aHit = (*theCAFI)[j];
      CLHEP::Hep3Vector hitPoint = aHit->getEntry();
      double   zz    = hitPoint.z();
      TheHistManager->GetHisto("zHits")->Fill(zz);
      if(tracklength0>8300.) TheHistManager->GetHisto("zHitsTrLoLe")->Fill(zz);
    }
    // varia = 0;
    //     if( varia == 0) {
    if( varia == 2) {


      int nhit11 = 0, nhit12 = 0, nhit13 = 0 ;
      double  totallosenergy= 0.;
      for (int j=0; j<theCAFI->entries(); j++) {
	BscG4Hit* aHit = (*theCAFI)[j];

	CLHEP::Hep3Vector hitEntryLocalPoint = aHit->getEntryLocalP();
	CLHEP::Hep3Vector hitExitLocalPoint = aHit->getExitLocalP();
	CLHEP::Hep3Vector hitPoint = aHit->getEntry();
	int trackIDhit  = aHit->getTrackID();
	unsigned int unitID = aHit->getUnitID();
	double  losenergy = aHit->getEnergyLoss();
	//double phi_hit   = hitPoint.phi();
	//if (phi_hit < 0.) phi_hit += twopi;

	double   zz    = hitPoint.z();

	TheHistManager->GetHisto("zHitsnoMI")->Fill(zz);

	if (verbosity > 2) {
	  std::cout << "BscTest:zHits = " << zz << std::endl;
	}

	themap[unitID] += losenergy;
	totallosenergy += losenergy;

	int zside, sector;
	BscNumberingScheme::unpackBscIndex(unitID);
	zside  = (unitID&32)>>5;
	sector = (unitID&7);

	//
	//=======================================
	G4ThreeVector middle = (hitExitLocalPoint+hitEntryLocalPoint)/2.;
	themapz[unitID]  = hitPoint.z()+middle.z();
	//=======================================
	// Y
	if(zside==1) {
	  //UserNtuples->fillg24(losenergy,1.);
	  if(losenergy > 0.00003) {
	    themap1[unitID] += 1.;
	  }
	}
	//X
	if(zside==2){
	  //UserNtuples->fillg25(losenergy,1.);
	  if(losenergy > 0.00005) {
	    themap1[unitID] += 1.;
	  }
	}
	//	   }
	//
	if(sector==1) {
	  nhit11 += 1;
	  //UserNtuples->fillg33(rr,1.);
	  //UserNtuples->fillg11(yy,1.);
	}
	if(sector==2) {
	  nhit12 += 1;
	  //UserNtuples->fillg34(rr,1.);
	  //UserNtuples->fillg86(yy,1.);
	}
	if(sector==3) {
	  nhit13 += 1;
	  //UserNtuples->fillg35(rr,1.);
	  //UserNtuples->fillg87(yy,1.);
	}

	if(lastpo.z()<z4  && lastpo.perp()< 120.) {
	  // MIonly:
	  //UserNtuples->fillg16(lastpo.z(),1.);
	  //UserNtuples->fillg18(zz,1.);
	  //                                                                     Station I
	  if( zz < z2){
	    //UserNtuples->fillg54(dx,1.);
	    //UserNtuples->fillg55(dy,1.);
	  }
	  //                                                                     Station II
	  if( zz < z3 && zz > z2){
	    //UserNtuples->fillg50(dx,1.);
	    //UserNtuples->fillg51(dy,1.);
	  }
	  //                                                                     Station III
	  if( zz < z4 && zz > z3){
	    //UserNtuples->fillg64(dx,1.);
	    //UserNtuples->fillg65(dy,1.);
	    //UserNtuples->filld209(xx,yy,1.);
	  }
	}
	else{
	  // no MIonly:
	  //UserNtuples->fillg17(lastpo.z(),1.);
	  //UserNtuples->fillg19(zz,1.);
	  //UserNtuples->fillg74(incidentEnergyHit,1.);
	  //UserNtuples->fillg75(float(trackIDhit),1.);
	  //                                                                     Station I
	  if( zz < z2){
	    //UserNtuples->fillg56(dx,1.);
	    //UserNtuples->fillg57(dy,1.);
	    //UserNtuples->fillg20(numofpart,1.);
	    //UserNtuples->fillg21(SumEnerDeposit,1.);
	    if(zside==1) {
	      //UserNtuples->fillg26(losenergy,1.);
	    }
	    if(zside==2){
	      //UserNtuples->fillg76(losenergy,1.);
	    }
	    if(trackIDhit == 1){
	      //UserNtuples->fillg70(dx,1.);
	      //UserNtuples->fillg71(incidentEnergyHit,1.);
	      //UserNtuples->fillg79(losenergy,1.);
	    }
	    else{
	      //UserNtuples->fillg82(dx,1.);
	    }
	  }
	  //                                                                     Station II
	  if( zz < z3 && zz > z2){
	    //UserNtuples->fillg52(dx,1.);
	    //UserNtuples->fillg53(dy,1.);
	    //UserNtuples->fillg22(numofpart,1.);
	    //UserNtuples->fillg23(SumEnerDeposit,1.);
	    //UserNtuples->fillg80(incidentEnergyHit,1.);
	    //UserNtuples->fillg81(float(trackIDhit),1.);
	    if(zside==1) {
	      //UserNtuples->fillg27(losenergy,1.);
	    }
	    if(zside==2){
	      //UserNtuples->fillg77(losenergy,1.);
	    }
	    if(trackIDhit == 1){
	      //UserNtuples->fillg72(dx,1.);
	      //UserNtuples->fillg73(incidentEnergyHit,1.);
	    }
	    else{
	      //UserNtuples->fillg83(dx,1.);
	    }
	  }
	  //                                                                     Station III
	  if( zz < z4 && zz > z3){
	    if(zside==1) {
	      //UserNtuples->fillg28(losenergy,1.);
	    }
	    if(zside==2){
	      //UserNtuples->fillg78(losenergy,1.);
	    }
	  }
	}
      }   // MIonly or noMIonly ENDED
      if(totallosenergy == 0.0) {
	std::cout << "BscTest:     number of hits = " << theCAFI->entries()   << std::endl;
	for (int j=0; j<theCAFI->entries(); j++) {
	  BscG4Hit* aHit = (*theCAFI)[j];
	  double  losenergy = aHit->getEnergyLoss();
	  std::cout << " j hits = " << j   << "losenergy = " << losenergy << std::endl;
	}
      }
      //   FIBRE Hit collected analysis
      double totalEnergy = 0.;
      int nhitsX = 0, nhitsY = 0, nsumhit = 0 ;
      for (int sector=1; sector<4; sector++) {
	int nhitsecX = 0, nhitsecY = 0;
	for (int zmodule=1; zmodule<11; zmodule++) {
	  for (int zside=1; zside<3; zside++) {
	    int det= 1;// nhit = 0;
	    //	int sScale = 20;
	    int index = BscNumberingScheme::packBscIndex(det, zside, sector);
	    double   theTotalEnergy = themap[index];
	    //   X planes
	    if(zside<2){ 
	      //UserNtuples->fillg47(theTotalEnergy,1.); 
	      if(theTotalEnergy > 0.00003) {
		nhitsX += 1;
		//		nhitsecX += themap1[index];
		//		nhit=1;
	      }
	    }
	    //   Y planes
	    else {
	      //UserNtuples->fillg49(theTotalEnergy,1.);
	      if(theTotalEnergy > 0.00005) {
		nhitsY += 1;
		//		nhitsecY += themap1[index];
		//		nhit=1;
	      }
	    }

	    totalEnergy += themap[index];
	  } // for
	} // for
          //UserNtuples->fillg39(nhitsecY,1.); 
	if(nhitsecX > 10 || nhitsecY > 10) {
	  nsumhit +=1;
	  //UserNtuples->fillp213(float(sector),float(1.),1.);
	}
	else{ //UserNtuples->fillp213(float(sector),float(0.),1.);
	}
      } // for

      if( nsumhit >=2 ) { //UserNtuples->fillp212(vy,float(1.),1.);
      }
      else{   //UserNtuples->fillp212(vy,float(0.),1.);
      }
    }   // MI or no MI or all  - end
  }                                                // primary end

  if (verbosity > 0) {
    std::cout << "BscTest:  END OF Event " << (*evt)()->GetEventID() << std::endl;
  }

}

