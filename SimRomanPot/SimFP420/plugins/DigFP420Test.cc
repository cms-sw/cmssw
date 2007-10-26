///////////////////////////////////////////////////////////////////////////////
// File: DigFP420Test 
// Date: 02.2007
// Description: DigFP420Test for FP420
// Modifications: std::  added wrt OSCAR code 
///////////////////////////////////////////////////////////////////////////////
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
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"

#include "SimRomanPot/SimFP420/interface/DigFP420Test.h"

#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

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

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <stdio.h>
//#include <gsl/gsl_fit.h>
//#include <gsl/gsl_cdf.h>


//================================================================
#include <cassert>

using namespace edm;
using namespace std;
///////////////////////////////////////////////////////////////////////////////

#define ddebugprim0
#define ddebugprim
#define ddebugprim1
//================================================================


enum ntfp420_elements {
  ntfp420_evt
};




//================================================================
DigFP420Test::DigFP420Test(const edm::ParameterSet & conf):conf_(conf),theDigitizerFP420(new DigitizerFP420(conf)){
  //constructor
  edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("DigFP420Test");
    verbosity    = m_Anal.getParameter<int>("Verbosity");
  //verbosity    = 1;

    fDataLabel  =   m_Anal.getParameter<std::string>("FDataLabel");
    fOutputFile =   m_Anal.getParameter<std::string>("FOutputFile");
    fRecreateFile = m_Anal.getParameter<std::string>("FRecreateFile");
    z420           = m_Anal.getParameter<double>("z420");
    zD2            = m_Anal.getParameter<double>("zD2");
    zD3            = m_Anal.getParameter<double>("zD3");
    sn0            =  m_Anal.getParameter<int>("NumberFP420Stations");
    pn0            =  m_Anal.getParameter<int>("NumberFP420SPlanes");
    dXXconst       = m_Anal.getParameter<double>("dXXFP420");//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7
    dYYconst       = m_Anal.getParameter<double>("dYYFP420");//  XSiDet/2. = 5.0
    ElectronPerADC = m_Anal.getParameter<double>("ElectronFP420PerAdc");
    xytype=2;

   
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   std::cout<<"============================================================================"<<std::endl;
   std::cout << "DigFP420Test constructor :: Initialized as observer"<< std::endl;
  }
	

//
	double zBlade = 5.00;
	double gapBlade = 1.6;
	ZSiPlane=2*(zBlade+gapBlade);

	double ZKapton = 0.1;
	ZSiStep=ZSiPlane+ZKapton;

	double ZBoundDet = 0.020;
	double ZSiElectr = 0.250;
	double ZCeramDet = 0.500;
//
	ZSiDetL = 0.250;
	ZSiDetR = 0.250;
	ZGapLDet= zBlade/2-(ZSiDetL+ZSiElectr+ZBoundDet+ZCeramDet/2);
//
  //  ZSiStation = 5*(2*(5.+1.6)+0.1)+2*6.+1.0 =  79.5  
	double ZSiStation = (pn0-1)*(2*(zBlade+gapBlade)+ZKapton)+2*6.+0.0;   // =  78.5  
  // 11.=e1, 12.=e2 in zzzrectangle.xml
	  double eee1=11.;
	  double eee2=12.;

	  zinibeg = (eee1-eee2)/2.;
  //////////////////////////zUnit = 8000.; // 2Stations
  //zD2 = 1000.;  // dist between centers of 1st and 2nd stations
  //zD3 = 8000.;  // dist between centers of 1st and 3rd stations
  //z420= 420000.;

  //                                                                                                                           .
  //                                                                                                                           .
  //  -300     -209.2             -150              -90.8                        0                                           +300
  //                                                                                                                           .
  //            X  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | X                        station                                          .
  //                   8*13.3+ 2*6 = 118.4                                    center                                           .
  //                                                                                                                           .

  // 10.mm -arbitrary to be right after end of Station 

  //  z1 = -150. + (118.4+10.)/2 + z420; // z1 - right after 1st Station

  z1 = zinibeg + (ZSiStation+10.)/2 + z420; // z1 - right after 1st Station






  z2 = z1+zD2;                       //z2 - right after 2nd Station
  z3 = z1+zD3;                       //z3 - right after 3rd   Station
  z4 = z1+2*zD3;
  //==================================
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   // std::cout << "DigFP420Test constructor :: Initialized as observer zUnit=" << zUnit << std::endl;
   std::cout << "DigFP420Test constructor :: Initialized as observer zD2=" << zD2 << std::endl;
   std::cout << " zD3=" << zD3 << std::endl;
   std::cout << " z1=" << z1 << " z2=" << z2 << " z3=" << z3 << " z4=" << z4 << std::endl;
   std::cout<<"============================================================================"<<std::endl;
  }
  //==================================

  // fp420eventntuple = new TNtuple("NTfp420event","NTfp420event","evt");
  //==================================

  whichevent = 0;

  //   fDataLabel      = "defaultData";
  //       fOutputFile     = "TheAnlysis.root";
  //       fRecreateFile   = "RECREATE";

        TheHistManager = new Fp420AnalysisHistManager(fDataLabel);

  //==================================
  if (verbosity > 0) {
   std::cout << "DigFP420Test constructor :: Initialized Fp420AnalysisHistManager"<< std::endl;
  }
  //==================================
  //sn0 = 4;// related to  number of station: sn0=3 mean 2 Stations
  //pn0 = 9;// related to number of planes: pn0=11 mean 10 Planes
  //-------------------------------------------------
  //-------------------------------------------------
  //-------------------------------------------------
    UseHalfPitchShiftInX_= true;
  //UseHalfPitchShiftInX_= false;
  
    UseHalfPitchShiftInY_= true;
  //UseHalfPitchShiftInY_= false;
  
  //-------------------------------------------------
    UseHalfPitchShiftInXW_= true;
  //UseHalfPitchShiftInXW_= false;
  
    UseHalfPitchShiftInYW_= true;
  //UseHalfPitchShiftInYW_= false;
  
  //-------------------------------------------------
  //-------------------------------------------------
  //-------------------------------------------------
  //-------------------------------------------------
	ldriftX= 0.050;
	ldriftY= 0.050;// was 0.040
	
	pitchX= 0.050;
	pitchY= 0.050;// was 0.040
	pitchXW= 0.400;
	pitchYW= 0.400;// was 0.040
	
	numStripsY = 201;        // Y plate number of strips:200*0.050=10mm (zside=1)
	numStripsX = 401;        // X plate number of strips:400*0.050=20mm (zside=2)
	numStripsYW = 51;        // Y plate number of W strips:50 *0.400=20mm (zside=1) - W have ortogonal projection
	numStripsXW = 26;        // X plate number of W strips:25 *0.400=10mm (zside=2) - W have ortogonal projection
	

	//  BoxYshft = [gap]+[dYcopper]+[dYsteel] = +12.3 + 0.05 + 0.15 = 12.5  ;  dYGap   =      0.2 mm
	//  dXXconst = 12.7+0.05;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7+0.05
        //	dXXconst = 12.7;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7

	//	dXXconst = 4.7;                     // gap = 4.3 instead 12.3
	//	dYYconst = 5.;// XSiDet/2.

	// change also in FP420ClusterMain.cc  CluFP420Test.cc         SimRomanPot/SimFP420/src/FP420DigiMain.cc  DigFP420Test.cc
	//ENC = 1800;
	//ENC = 3000;
	//ENC = 2160;
	ENC = 960;

	//	ElectronPerADC =300;
	Thick300 = 0.300;
//

  // Initialization:

	theFP420NumberingScheme = new FP420NumberingScheme();
	//	theDigitizerFP420 = new DigitizerFP420(conf_);
	//theClusterizerFP420 = new ClusterizerFP420(conf_);
	//theTrackerizerFP420 = new TrackerizerFP420(conf_);
  std::cout<<"DigFP420Test: End of Initialization processes"<<std::endl;
//
}



DigFP420Test::~DigFP420Test() {
  //  delete UserNtuples;
  delete theFP420NumberingScheme;
  delete theDigitizerFP420;
//  delete theClusterizerFP420;
//  delete theTrackerizerFP420;
  
  std::cout << "DigFP420Test create output root file:";

//  TFile fp420OutputFile("newntfp420.root","RECREATE");
//  std::cout << "DigFP420Test output root file has been created";
//  fp420eventntuple->Write();
//  std::cout << ", written";
//  fp420OutputFile.Close();
//  std::cout << ", closed";
//  delete fp420eventntuple;
//  std::cout << ", and deleted" << std::endl;
  
  //------->while end
  
  // Write histograms to file
  TheHistManager->WriteToFile(fOutputFile,fRecreateFile);
  
  std::cout<<"DigFP420Test: End of process"<<std::endl;
  
  if (verbosity > 0) {
    std::cout << std::endl << "DigFP420Test Destructor  -------->  End of DigFP420Test : "
	      << std::cout << std::endl; 
  }
  std::cout<<"DigFP420Test: RETURN"<<std::endl;
  
  
}

//================================================================

//================================================================
// Histoes:
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::Fp420AnalysisHistManager(TString managername)
{
        // The Constructor
        fTypeTitle=managername;
        fHistArray = new TObjArray();      // Array to store histos
        fHistNamesArray = new TObjArray(); // Array to store histos's names

	TH1::AddDirectory(kFALSE);
	//	fHistArray->SetDirectory(0);
        BookHistos();

        fHistArray->Compress();            // Removes empty space
        fHistNamesArray->Compress();

//      StoreWeights();                    // Store the weights

}
//-----------------------------------------------------------------------------

Fp420AnalysisHistManager::~Fp420AnalysisHistManager()
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

void Fp420AnalysisHistManager::BookHistos()
{
        // Book the histograms and add them to the array

  // at Start: (mm)
  double    yt1 = -3.,  yt2= 3.;   int kyt=6;
    HistInit("YVall",   "YVall",  kyt, yt1,yt2);
    HistInit("YVz1",    "YVz1",   kyt, yt1,yt2);
    HistInit("YV1z2",   "YV1z2",  kyt, yt1,yt2);
    HistInit("YVz2",    "YVz2",   kyt, yt1,yt2);
    HistInit("YVz3",    "YVz3",   kyt, yt1,yt2);
    HistInit("YVz4",    "YVz4",   kyt, yt1,yt2);
  double    xt1 =-100.,  xt2=100.;   int kxt= 100;
    HistInit("XVall",   "XVall",  kxt, xt1,xt2);
    HistInit("XVz1",    "XVz1",   kxt, xt1,xt2);
    HistInit("XV1z2",   "XV1z2",  kxt, xt1,xt2);
    HistInit("XVz2",    "XVz2",   kxt, xt1,xt2);
    HistInit("XVz3",    "XVz3",   kxt, xt1,xt2);
    HistInit("XVz4",    "XVz4",   kxt, xt1,xt2);



    HistInit("HitLosenergy",  "HitLosenergy",             100, 0.,0.00025);
    HistInit("Hithadrenergy",  "Hithadrenergy",           100, 0.,0.25);
    HistInit("HitIncidentEnergy",  "HitIncidentEnergy",   100, 0.,7000000.);
    HistInit("HitTimeSlice",  "HitTimeSlice",             100, 0.,200.);
    HistInit("HitEnergyDeposit",  "HitEnergyDeposit",     100, 0.,0.25);
    HistInit("HitPabs",  "HitPabs",                       100, 0.,7000.);
    HistInit("HitTof",  "HitTof",                         100, 0.,3100.);
    HistInit("HitParticleType",  "HitParticleType",       100, -2500,2500.);
    HistInit("HitThetaAtEntry",  "HitThetaAtEntry",       100, 0.0005,0.0035);
    HistInit("HitPhiAtEntry",  "HitPhiAtEntry",           100, -190,190.);
    HistInit("HitX",  "HitX",   100, -30.,0.);
    HistInit("HitY",  "HitY",   100, -5.,5.);
    HistInit("HitZ",  "HitZ",   100, 419000.,429000.);
    HistInit("HitParentId",  "HitParentId",   100, 0.,500.);
    HistInit("HitVx",  "HitVx",   100, -50.,0.);
    HistInit("HitVy",  "HitVy",   100, -16,+16.);
    HistInit("HitVz",  "HitVz",   100, -500000,500000.);

    HistInit("HitIncidentEnergyNoMI",  "HitIncidentEnergy NoMI",   100, 0.,7000000.);
    HistInit("HitIncidentEnergyMI",  "HitIncidentEnergy MI",       100, 0.,7000000.);

    HistInit("HitLosenergyH",  "HitLosenergyH",             100, 0.,0.00025);
    HistInit("HithadrenergyH",  "HithadrenergyH",           100, 0.,0.25);
    HistInit("HitIncidentEnergyH",  "HitIncidentEnergyH",   100, 0.,7000000.);
    HistInit("HitTimeSliceH",  "HitTimeSliceH",             100, 0.,200.);
    HistInit("HitEnergyDepositH",  "HitEnergyDepositH",     100, 0.,0.25);
    HistInit("HitPabsH",  "HitPabsH",                       100, 0.,7000.);
    HistInit("HitTofH",  "HitTofH",                         100, 0.,3100.);
    HistInit("HitParticleTypeH",  "HitParticleTypeH",       100, -2500,2500.);
    HistInit("HitThetaAtEntryH",  "HitThetaAtEntryH",       100, 0.0005,0.0035);
    HistInit("HitPhiAtEntryH",  "HitPhiAtEntryH",           100, -190,190.);
    HistInit("HitXH",  "HitXH",   100, -30.,0.);
    HistInit("HitYH",  "HitYH",   100, -5.,5.);
    HistInit("HitZH",  "HitZH",   100, 419000.,429000.);
    HistInit("HitParentIdH",  "HitParentIdH",   100, 0.,500.);
    HistInit("HitVxH",  "HitVxH",   100, -50.,0.);
    HistInit("HitVyH",  "HitVyH",   100, -16,+16.);
    HistInit("HitVzH",  "HitVzH",   100, -500000,500000.);

    HistInit("SIDETLenDep",  "SIDETLenDep",             100, 0.,0.25);
    HistInit("SIDETRenDep",  "SIDETRenDep",             100, 0.,0.25);




    HistInit("NumofpartNoMI", "Numofpart No MI",       100,   0.,20. );
    HistInit("NumofpartOnlyMI", "Numofpart Only MI",   100,   0.,20. );

    HistInit("NumberHitsNoMI", "Number Hits No MI",       110,   0.,110. );
    HistInit("NumberHitsOnlyMI", "Number Hits Only MI",   110,   0.,110. );
    HistInit("NumberHitsFinal", "Number Hits Final",      110,   0.,110. );
    HistInit("NHitsAll", "N Hits All",                    110,   0.,110. );
    HistInit("NumofHitsSec1", "NumofHitsSec1",            11,   0.,11. );
    HistInit("NumofHitsSec2", "NumofHitsSec2",            11,   0.,11. );
    HistInit("NumofHitsSec3", "NumofHitsSec3",            11,   0.,11. );


    HistInit("PrimaryIDMom",  "Primary ID Mom",       100,   0.,  -7000000. );
    HistInit("PrimaryMom",  "Primary Mom",       100,   0.,  7000000. );
    HistInit("PrimaryXi0",  "Primary Xi0",       100,   0.000000001,    0.1 );
    HistInit("PrimaryXi",   "Primary Xi",        100,   0.00001 ,   0.1);
    HistInit("PrimaryXiLog","Primary Xi Log",    100,       -6.,    -1.);
    HistInit("PrimaryEta",  "Primary Eta",        50,        8.,    13.);


    HistInit("PrimaryPhigrad", "Primary Phigrad",    100,   0.,360. );
    HistInit("PrimaryTh",      "Primary Th",         100,   0.,-0.5 );

    HistInit("PrimaryLastpoZ0", "Primary Lastpo Z0",   100, 410000.,430000. );
    HistInit("numofpart0", "numofpart0",   100, 0.,-1. );

    HistInit("PrimaryLastpoZ", "Primary Lastpo Z",   100, 420000.,430000. );
    HistInit("PrimaryLastpoX", "Primary Lastpo X Z<z4",   100, -30., 30. );
    HistInit("PrimaryLastpoY", "Primary Lastpo Y Z<z4",   100, -30., 30. );
    HistInit("XLastpoNumofpart", "Primary Lastpo X n>10",   100, 0.,-1. );
    HistInit("YLastpoNumofpart", "Primary Lastpo Y n>10",   100, 0.,-1. );


    HistInit("PrimaryIDMom2",  "Primary ID Mom2",       100,   0.,  -7000000. );
    HistInit("PrimaryMom2",  "Primary Mom2",       100,   0.,  7000000. );
    HistInit("PrimaryTh2",      "Primary Th2",     100,   0.,         -0.5 );
    HistInit("PrimaryPhigrad2", "Primary Phigrad2",100,   0.,         360. );
    HistInit("PrimaryEta2",  "Primary Eta2",        50,   8.,           13.);
    HistInit("PrimaryXi2",   "Primary Xi2",        100,   0.00001 ,   0.1);

    HistInit("PrimaryIDMom3",  "Primary ID Mom3",       100,   0.,  -7000000. );
    HistInit("PrimaryMom3",  "Primary Mom3",       100,   0.,  7000000. );
    HistInit("PrimaryTh3",      "Primary Th3",     100,   0.,         -0.5 );
    HistInit("PrimaryPhigrad3", "Primary Phigrad3",100,   0.,         360. );
    HistInit("PrimaryEta3",  "Primary Eta3",        50,   8.,           13.);
    HistInit("PrimaryXi3",   "Primary Xi3",        100,   0.00001 ,   0.1);




    HistInit("VtxX0", "Vtx X0",       75, -18.,-3. );
    HistInit("VtxX", "Vtx X",       100, -20.,+5. );
    HistInit("VtxY", "Vtx Y",       100, -1., 1. );
    HistInit("VtxZ", "Vtx Z",       100, 410000., 430000. );

    HistInit("2Dxy1", "2Dxy 1",   100, -70., 70.,100, -70., 70. );
    HistInit("2Dxz1", "2Dxz 1",   100, -50., 50.,200, 419000.,+429000. );
    HistInit("XenDep", "XenDep",   100, -100.,+100.);
    HistInit("YenDep", "YenDep",   100, -100.,+100.);
    HistInit("ZenDep", "ZenDep",   300, 410000.,+440000.);
          // Digis
//     int nx=201; float xn=nx; int mx=100; float xm=50.;
//     int ny=101; float yn=ny; int my=100; float ym=50.;
    int nx=401; float xn=nx; int mx=40; float xm=25.;
    int nxw=26; float xnw=nxw;
    HistInit("DigiXstrip",    "Digi Xstrip ",      nx,   0.,xn );
    HistInit("DigiXWstrip",    "Digi XWstrip ",      nxw,   0.,xnw );
    HistInit("2DigiXXW","2Digi X XW",      nxw,   0.,xnw, nx,   0.,xn );
    HistInit("2DigiXXWAmplitude","2Digi X XWA",      nxw,   0.,xnw, nx,   0.,xn );
    HistInit("DigiXstripAdc", "Digi Xstrip Adc",   100,   1.,101. );
    HistInit("AmplitudeX", "Amplitude X",          100,   1.,101. );
    HistInit("AmplitudeXW", "Amplitude XW",        100,   1.,101. );
    HistInit("DigiAmplitudeX", "Digi Amplitude X",      nx,   0.,xn );
    HistInit("DigiAmplitudeXW", "Digi Amplitude XW",      nxw,   0.,xnw );
    HistInit("DigiXstripAdcSigma",  "Digi Xstrip Adc in SigmaNoise",        mx,   0.,xm  );
    HistInit("DigiXWstripAdcSigma",  "Digi XWstrip Adc in SigmaNoise",        mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma1",  "Digi Xstrip Adc in SigmaNoise1",      mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma2",  "Digi Xstrip Adc in SigmaNoise2",      mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma3",  "Digi Xstrip Adc in SigmaNoise3",      mx,   0.,xm  );

    int ny=201; float yn=ny; int my=40; float ym=25.;
    int nyw=51; float ynw=nyw;
    HistInit("DigiYWstrip",    "Digi YWstrip ",      nyw,   0.,ynw );
    HistInit("2DigiYYW","2Digi Y YW",      nyw,   0.,ynw, ny,   0.,yn );
    HistInit("2DigiYYWAmplitude","2Digi Y YWA",      nyw,   0.,ynw, ny,   0.,yn );
    HistInit("DigiYstrip",    "Digi Ystrip ",      ny,   0.,yn );
    HistInit("DigiYstripAdc", "Digi Ystrip Adc",   100,   1.,101. );
    HistInit("AmplitudeY", "Amplitude Y",          100,   1.,101. );
    HistInit("AmplitudeYW", "Amplitude YW",        100,   1.,101. );
    HistInit("DigiAmplitudeY", "Digi Amplitude Y",      ny,   0.,yn );
    HistInit("DigiAmplitudeYW", "Digi Amplitude YW",      nyw,   0.,ynw );
    HistInit("DigiYstripAdcSigma",  "Digi Ystrip Adc in SigmaNoise",        my,   0.,ym );

	 // Clusters:
    HistInit("xref",     "xref",        10,  -24.5,-4.5);
    HistInit("xrefNoMI", "xrefNoMI",    10,  -24.5,-4.5);
    HistInit("xref2NoMI", "xref2NoMI",  10,  -24.5,-4.5);
    HistInit("xref2MI", "xref2MI",      10,  -24.5,-4.5);
    HistInit("xrefMI",   "xrefMI",      10,  -24.5,-4.5);
    HistInit("xrefAcc",  "xrefAcc",     10,  -24.5,-4.5);

    HistInit("xref2",    "xref2",       10,  -24.5,-4.5);

    HistInit("yref",     "yref",        10,  -5.,5.);
    HistInit("yrefNoMI", "yrefNoMI",    10,  -5.,5.);
    HistInit("yrefMI",   "yrefMI",      10,  -5.,5.);
    HistInit("yrefAcc",  "yrefAcc",     10,  -5.,5.);

    HistInit("yref2",    "yref2",       10,  -5.,5.);



//
//
    HistInit("ATest", "ATest",  100,  0.,100);

//
//
    HistInit("ZZZall", "ZZZall",  100,  420000.,0.);
    HistInit("ZZZ420", "ZZZ420",  100,  420000.,0.);
    HistInit("XXX420", "XXX420",  100,  -20.,20);
    HistInit("YYY420", "YYY420",  100,  -10.,10);
    HistInit("npart420", "npart420",  10,  0.,10.);

    HistInit("2DXY420", "2DXY420",    100, -25.,5.,100, -5.,5.);
    HistInit("2DXY420refLast", "2DXY420refLast",100, -25.,5.,100, -5.,5.);
    HistInit("2DXY420refBeg", "2DXY420refBeg",100, -25.,5.,100, -5.,5.);

//
    HistInit("EntryX", "EntryX",100, 5.,-5.);
    HistInit("EntryY", "EntryY",100, 5.,-5.);
    HistInit("midZ", "midZ",100, 5.,-5.);

    HistInit("EntryXH", "EntryXH",100, 5.,-5.);
    HistInit("EntryYH", "EntryYH",100, 5.,-5.);
    HistInit("midZH", "midZH",100, 5.,-5.);


    HistInit("EntryZ1", "EntryZ1",100, 5.,-5.);
    HistInit("EntryZ2", "EntryZ2",100, 5.,-5.);
    HistInit("EntryZ3", "EntryZ3",100, 5.,-5.);
    HistInit("EntryZ4", "EntryZ4",100, 5.,-5.);


//
    // 2D:
//    HistInit("2DSecVsR",   "2DSecVsR",   100, 0.,20.,100, 0.,100.);
//    HistInit("2DSecVsZ",   "2DSecVsZ",   100, 0.,20.,100, 410000.,460000.);
//    HistInit("2DSecVsHits","2DSecVsHits",100, 0.,16.,110, 0.,110.);
//    HistInit("2DHitsVsR",  "2DHitsVsR",  100, 0.,100.,110, 0.,110.);
//    HistInit("2DHitsVsZ",  "2DHitsVsZ",  100, 0.,100.,100, 419000.,460000.);


    HistInit("icurtrack", "icurtrack",  10, 0.,  5.);
    HistInit("nvertexa", "nvertex",      10, 0,  5);

//
//
//
}

//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::WriteToFile(TString fOutputFile,TString fRecreateFile)
{

        //Write to file = fOutputFile

        std::cout <<"================================================================"<<std::endl;
        std::cout <<" Write this Analysis to File "<<fOutputFile<<std::endl;
        std::cout <<"================================================================"<<std::endl;
        TFile* file = new TFile(fOutputFile, fRecreateFile);
        std::cout <<" new TFile DONE with"<< fRecreateFile << std::endl;

        fHistArray->Write();
        std::cout <<" Write DONE "<< std::endl;

        file->Close();
        std::cout <<" Close DONE "<< std::endl;
       // The Destructor

}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup)
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

void Fp420AnalysisHistManager::HistInit(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)
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

TH1F* Fp420AnalysisHistManager::GetHisto(Int_t Number)
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

TH2F* Fp420AnalysisHistManager::GetHisto2(Int_t Number)
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

TH1F* Fp420AnalysisHistManager::GetHisto(const TObjString histname)
{
        // Get a histogram from the array with name = histname

        Int_t index = fHistNamesArray->IndexOf(&histname);
        return GetHisto(index);
}
//-----------------------------------------------------------------------------

TH2F* Fp420AnalysisHistManager::GetHisto2(const TObjString histname)
{
        // Get a histogram from the array with name = histname

        Int_t index = fHistNamesArray->IndexOf(&histname);
        return GetHisto2(index);
}
//-----------------------------------------------------------------------------

void Fp420AnalysisHistManager::StoreWeights()
{
        // Add structure to each histogram to store the weights

        for(int i = 0; i < fHistArray->GetEntries(); i++){
                ((TH1F*)(fHistArray->At(i)))->Sumw2();
        }
}
// Histoes end :



//==================================================================== per JOB
void DigFP420Test::update(const BeginOfJob * job) {
  //job
  std::cout<<"DigFP420Test:beggining of job"<<std::endl;;
}


//==================================================================== per RUN
void DigFP420Test::update(const BeginOfRun * run) {
  //run

 std::cout << std::endl << "DigFP420Test:: Begining of Run"<< std::endl; 
}


void DigFP420Test::update(const EndOfRun * run) {;}



//=================================================================== per EVENT
void DigFP420Test::update(const BeginOfEvent * evt) {
  iev = (*evt)()->GetEventID();
#ifdef ddebugprim
    std::cout <<"DigFP420Test:: ==============Event number = " << iev << std::endl;
#endif
    std::cout <<"DigFP420Test:: ==============Event number = " << iev << std::endl;
  whichevent++;



}

//=================================================================== per Track
void DigFP420Test::update(const BeginOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  G4ThreeVector   track_mom  = (*trk)()->GetMomentum();
#ifdef ddebugprim
//    std::cout <<" DigFP420Test::=======BeginOfTrack number = " << itrk << std::endl;
#endif
//  if(itrk == 1) {
  if(track_mom.z() > 100000.) {
     SumEnerDeposit = 0.;
     SumEnerDeposit1 = 0.;
     numofpart = 0;
     SumStepl = 0.;
     SumStepc = 0.;
  }
}



//=================================================================== per EndOfTrack
void DigFP420Test::update(const EndOfTrack * trk) {
  itrk = (*trk)()->GetTrackID();
  
  G4ThreeVector   track_mom  = (*trk)()->GetMomentum();
  G4String       particleType   = (*trk)()->GetDefinition()->GetParticleName();   //   !!!
#ifdef ddebugprim
  if (iev==7267) {
  G4int         parentID       = (*trk)()->GetParentID();   //   !!!
  G4TrackStatus   trackstatus    = (*trk)()->GetTrackStatus();   //   !!!
  G4double       entot          = (*trk)()->GetTotalEnergy();   //   !!! deposited on step
  G4int         curstepnumber  = (*trk)()->GetCurrentStepNumber();
  std::cout <<" ==========EndOfTrack number = " << itrk << std::endl;
  std::cout <<" sum dep. energy over all steps along primary track = " << SumEnerDeposit << std::endl;
  std::cout <<" TrackLength= " << (*trk)()->GetTrackLength() << std::endl;
  std::cout <<" GetTrackID= " << (*trk)()->GetTrackID() << std::endl;
  std::cout <<" GetMomentum= " << track_mom << std::endl;
  std::cout <<" particleType= " << particleType << std::endl;
  std::cout <<" parentID= " << parentID << std::endl;
  std::cout <<" trackstatus= " << trackstatus << std::endl;
  std::cout <<" entot= " << entot << std::endl;
  std::cout <<" curstepnumber= " << curstepnumber << std::endl;
  }
#endif
//  if(itrk == 1) {
  if(track_mom.z() > 100000.) {
    G4double tracklength  = (*trk)()->GetTrackLength();    // Accumulated track length
    G4ThreeVector   vert_mom  = (*trk)()->GetVertexMomentumDirection();
    G4ThreeVector   vert_pos  = (*trk)()->GetVertexPosition(); // vertex ,where this track was created

  //float eta = 0.5 * log( (1.+vert_mom.z()) / (1.-vert_mom.z()) );
    float phi = atan2(vert_mom.y(),vert_mom.x());
    if (phi < 0.) phi += twopi;
    //float phigrad = phi*180./pi;

    float XV = vert_pos.x(); // mm
    float YV = vert_pos.y(); // mm
      //UserNtuples->fillg543(phigrad,1.);
      //UserNtuples->fillp203(phigrad,SumStepl,1.);
      //UserNtuples->fillp201(XV,SumStepl,1.);
    TheHistManager->GetHisto("XVall")->Fill(XV);
    TheHistManager->GetHisto("YVall")->Fill(YV);
// MI = (multiple interactions):
       if(tracklength < z4) {
       }

        // last step information
        const G4Step* aStep = (*trk)()->GetStep();
        //   G4int csn = (*trk)()->GetCurrentStepNumber();
        //   G4double sl = (*trk)()->GetStepLength();
         // preStep
         G4StepPoint*      preStepPoint = aStep->GetPreStepPoint(); 
         lastpo   = preStepPoint->GetPosition();	

	 // Analysis:
	 if(lastpo.z()<z1 && lastpo.perp()< 100.) {
             //UserNtuples->fillg525(eta,1.);
	   TheHistManager->GetHisto("XVz1")->Fill(XV);
	   TheHistManager->GetHisto("YVz1")->Fill(YV);
             //UserNtuples->fillg556(phigrad,1.);
         }
	 if((lastpo.z()>z1 && lastpo.z()<z2) && lastpo.perp()< 100.) {
             //UserNtuples->fillg526(eta,1.);
	   TheHistManager->GetHisto("XV1z2")->Fill(XV);
	   TheHistManager->GetHisto("YV1z2")->Fill(YV);
             //UserNtuples->fillg557(phigrad,1.);
         }
	 if(lastpo.z()<z2 && lastpo.perp()< 100.) {
             //UserNtuples->fillg527(eta,1.);
	   TheHistManager->GetHisto("XVz2")->Fill(XV);
	   TheHistManager->GetHisto("YVz2")->Fill(YV);
              //UserNtuples->fillg558(phigrad,1.);
         //UserNtuples->fillg521(lastpo.x(),1.);
         //UserNtuples->fillg522(lastpo.y(),1.);
         //UserNtuples->fillg523(lastpo.z(),1.);
        }
	 if(lastpo.z()<z3 && lastpo.perp()< 100.) {
             //UserNtuples->fillg528(eta,1.);
	   TheHistManager->GetHisto("XVz3")->Fill(XV);
	   TheHistManager->GetHisto("YVz3")->Fill(YV);
             //UserNtuples->fillg559(phigrad,1.);
         }
	 if(lastpo.z()<z4 && lastpo.perp()< 100.) {
             //UserNtuples->fillg529(eta,1.);
	   TheHistManager->GetHisto("XVz4")->Fill(XV);
	   TheHistManager->GetHisto("YVz4")->Fill(YV);
         }


  }//if(itrk == 1
}

// =====================================================================================================

//=================================================================== each STEP
void DigFP420Test::update(const G4Step * aStep) {
// ==========================================================================
  
  // track on aStep:                                                                                         !
  G4Track*     theTrack     = aStep->GetTrack();   
  TrackInformation* trkInfo = dynamic_cast<TrackInformation*> (theTrack->GetUserInformation());
   if (trkInfo == 0) {
     std::cout << "DigFP420Test on aStep: No trk info !!!! abort " << std::endl;
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
  
  //  double costheta =vert_mom.z()/sqrt(vert_mom.x()*vert_mom.x()+vert_mom.y()*vert_mom.y()+vert_mom.z()*vert_mom.z());
  //  double theta = acos(min(max(costheta,double(-1.)),double(1.)));
  //  float eta = -log(tan(theta/2));
  //  double phi = -1000.;
  //  if (vert_mom.x() != 0) phi = atan2(vert_mom.y(),vert_mom.x()); 
  //  if (phi < 0.) phi += twopi;
  //  double phigrad = phi*360./twopi;  

#ifdef ddebug
  if (iev==7267) {
     std::cout << " ====================================================================" << std::endl;
     std::cout << " ==========================================111111" << std::endl;
     std::cout << "DigFP420Test on aStep: Entered for track ID=" << id 
          << " ID Name= " << particleType
          << " at stepNumber= " << curstepnumber 
          << " ID onCaloSur..= " << trkInfo->getIDonCaloSurface()
          << " CaloID Check= " << trkInfo->caloIDChecked() 
          << " trackstatus= " << trackstatus
          << " trackmom= " << trackmom
          << " entot= " << entot
          << " vert_where_track_created= " << vert_pos
          << " vert_mom= " << vert_mom
       //          << " Accumulated tracklength= " << tracklength
          << " parent ID = " << parentID << std::endl;
  G4ProcessManager* pm   = theTrack->GetDefinition()->GetProcessManager();
  G4ProcessVector* pv = pm->GetProcessList();
 G4int np = pm->GetProcessListLength();
 for(G4int i=0; i<np; i++) {
 std::cout <<"i=   " <<i << "ProcessName = "  << ((*pv)[i])->GetProcessName() << std::endl;
   }
  }
#endif


  // step points:                                                                                         !
  //G4double        stepl         = aStep->GetStepLength();
  G4double        EnerDeposit   = aStep->GetTotalEnergyDeposit();

  // preStep
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
#ifdef ddebug
      float th_tr     = preposition.theta();
      float eta_tr    = -log(tan(th_tr/2));
      float phi_tr    = preposition.phi();
      if (phi_tr < 0.) phi_tr += twopi;

     std::cout << "============aStep: information:============" << std::endl;
     std::cout << " EneryDeposited = " << EnerDeposit
          << " stepl = "          << stepl << std::endl;

     std::cout << "============preStep: information:============" << std::endl;
     std::cout << " preposition = "    << preposition
          << " prelocalpoint = "  << prelocalpoint
          << " eta_tr = "         << eta_tr
          << " phi_tr = "         << phi_tr*360./twopi
          << " prevolume = "      << prename
//          << " posvolume = "      << aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName()
          << " pre_levels = "     << pre_levels
          << std::endl;
      if (pre_levels > 0) {
        for (int i1=0; i1<pre_levels; i1++) 
          std::cout << "level= " << i1 << "name= " << name1[i1] << "copy= " << copyno1[i1] << std::endl;
      }

#endif
      if ( id == 1 ) {
	// on 1-st step:
	if (curstepnumber == 1 ) {
	  entot0 = entot;
	}
	
      // deposition of energy on steps along primary track
      // collect sum deposited energy on all steps along primary track
	//	SumEnerDeposit += EnerDeposit;
	// position of step for primary track:
	TheHistManager->GetHisto("XenDep")->Fill(preposition.x(),EnerDeposit);
	TheHistManager->GetHisto("YenDep")->Fill(preposition.y(),EnerDeposit);
	TheHistManager->GetHisto("ZenDep")->Fill(preposition.z(),EnerDeposit);

	TheHistManager->GetHisto2("2Dxy1")->Fill(preposition.x(),preposition.y(),EnerDeposit);
	TheHistManager->GetHisto2("2Dxz1")->Fill(preposition.x(),preposition.z(),EnerDeposit);
	// last step of primary track
	if (trackstatus == 2 ) {
          tracklength0 = tracklength;// primary track length 
	}

      }//     if ( id == 1
      // end of primary track !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      if (trackstatus != 2 ) {
	// for SD:   Si Det.:   SISTATION:SIPLANE:(SIDETL+BOUNDDET        +SIDETR + CERAMDET)
	//	if(prename == "SIDETL" || prename == "SIDETR" ) {
	if(prename == "SIDETL") {
	  SumEnerDeposit += EnerDeposit;
	  // last step of current SD Volume:
	  G4String posname = aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName();
	  if(posname != "SIDETL") {
	    if(SumEnerDeposit != 0.0) TheHistManager->GetHisto("SIDETLenDep")->Fill(SumEnerDeposit);
	    SumEnerDeposit = 0.;
	  }
	}
	
	
	if(prename == "SIDETR") {
	  SumEnerDeposit1 += EnerDeposit;
	  // last step of current SD Volume:
	  G4String posname = aStep->GetPostStepPoint()->GetPhysicalVolume()->GetName();
	  if(posname != "SIDETR") {
	    if(SumEnerDeposit1 != 0.0) TheHistManager->GetHisto("SIDETLenDep")->Fill(SumEnerDeposit1);
	    SumEnerDeposit1 = 0.;
	  }








	  if(EnerDeposit != 0.0) TheHistManager->GetHisto("SIDETRenDep")->Fill(EnerDeposit);
	}
	
      }//if (tracksta
      



      // particles deposit their energy along primary track
      if (parentID == 1 && curstepnumber == 1) {
	// if(trackmom.z() > 100000. && curstepnumber == 1) {
	numofpart += 1;
      }// if (parentID == 1 && curstepnumber == 1)
      
      
  // ==========================================================================
}
// ==========================================================================
// ==========================================================================
int DigFP420Test::detLevels(const G4VTouchable* touch) const {

  //Return number of levels
  if (touch) 
    return ((touch->GetHistoryDepth())+1);
  else
    return 0;
}
// ==========================================================================

G4String DigFP420Test::detName(const G4VTouchable* touch, int level,
                                    int currentlevel) const {

  //Go down to current level
  if (level > 0 && level >= currentlevel) {
    int ii = level - currentlevel; 
    return touch->GetVolume(ii)->GetName();
  } else {
    return "NotFound";
  }
}

void DigFP420Test::detectorLevel(const G4VTouchable* touch, int& level,
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
void DigFP420Test::update(const EndOfEvent * evt) {
  // ==========================================================================
  
  // Fill-in ntuple
  fp420eventarray[ntfp420_evt] = (float)whichevent;
  
  //
  // int trackID = 0;
  G4PrimaryParticle* thePrim=0;
  G4double vz=-99990.;
  G4double vx=-99990.,vy=-99990.;
  
  
#ifdef ddebugprim1
      std::cout << " -------------------------------------------------------------" << std::endl;
      std::cout << " -------------------------------------------------------------" << std::endl;
      std::cout << " -------------------------------------------------------------" << std::endl;
#endif
  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  
#ifdef ddebugprim1
  if (nvertex !=1) std::cout << "DigFP420Test:NumberOfPrimaryVertex != 1 --> = " << nvertex<<std::endl;
  std::cout << "NumberOfPrimaryVertex:" << nvertex << std::endl;
#endif
  int varia= 0,varia2= 0,varia3= 0;   // = 0 -all; =1 - MI; =2 - noMI
  double phi= -100.,  phigrad= -100.,  th= -100.,  eta= -100.,  xi= -100.; 
  double phi2= -100., phigrad2= -100., th2= -100., eta2= -100.,  xi2= -100.; 
  double phi3= -100., phigrad3= -100., th3= -100., eta3= -100.,  xi3= -100.; 
  double zmilimit = -100.;
  double XXX420   = -100.;
  double YYY420   = -100.;
  //if(zUnit==4000.) zmilimit= z3;
  //if(zUnit==8000.) zmilimit= z2;
  zmilimit= z3;// last variant
#ifdef ddebugprim1
  std::cout << "zmilimit= :" << zmilimit << std::endl;
#endif
// ==========================================================================================loop over vertexies
  double zref=-100., xref=-100., yref=-100., bxtrue=-100., bytrue=-100.,dref12=-100.,drefy12=-100.,ppmom=-100.;
    //ref = z1+8000.;     // info: center of 1st station at 0.
  double       xref2=-100., yref2=-100., bxtrue2=-100., bytrue2=-100.,ppmom2=-100.;
  double       xref3=-100., yref3=-100., bxtrue3=-100., bytrue3=-100.,ppmom3=-100.;
  double ZZZ420=-999999;
  double zbegin = 420000. - ZZZ420 ;
  double zfinis = 428000. - ZZZ420 ;
  double  zref1 =    8000. ;     // zref1 is z of measurement base of the detector
  int idmom=0, idmom2=0, idmom3=0 , icurtrack = 0;

#ifdef ddebugprim1
      std::cout << "nvertex =  " << nvertex << std::endl;
#endif
      TheHistManager->GetHisto("nvertexa")->Fill(nvertex);

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  for (int iv = 0 ; iv<nvertex; ++iv) {
    // loop over vertexes
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(iv);
    if (avertex == 0) std::cout<<"DigFP420Test:End Of Event ERR: pointer to vertex = 0"<< std::endl;
    G4int npart = avertex->GetNumberOfParticle();

    TheHistManager->GetHisto("ZZZall")->Fill(avertex->GetZ0());
    


#ifdef ddebugprim1
      std::cout << " ================================================================= iv=" << iv << std::endl;
      std::cout << "nparticles =  " <<avertex->GetNumberOfParticle() << std::endl;
      std::cout << "Z0 =  " <<avertex->GetZ0() << std::endl;
      std::cout << "X0 =  " <<avertex->GetX0() << std::endl;
      std::cout << "Y0 =  " <<avertex->GetY0() << std::endl;
#endif
    if(avertex->GetZ0() < 400000.){
      // vertex <400 m
      // temporary:
      // if(npart==1) {
      //	G4ThreeVector   mom  = avertex->GetPrimary(0)->GetMomentum();
      //	if(mom.z()<-100000.){
      //	  eta0 = -log(tan(mom.theta()/2));
      //	  eta0 = -eta0;
      //	  xi0 = 1.-mom.mag()/7000000.;
      //	}
      // }
    }
    // =======================================================over ZZZ420 vertexies
    else{
      // vertex >400 m
#ifdef ddebugprim1
      std::cout << "Vertex number :" <<iv << std::endl;
      std::cout << "Vertex Z= :" <<(*evt)()->GetPrimaryVertex(iv)->GetZ0() << std::endl;
      std::cout << "Vertex X= :" <<(*evt)()->GetPrimaryVertex(iv)->GetX0() << std::endl;
      std::cout << "Vertex Y= :" <<(*evt)()->GetPrimaryVertex(iv)->GetY0() << std::endl;
      
#endif
      XXX420 = avertex->GetX0();
      YYY420 = avertex->GetY0();
      ZZZ420 = avertex->GetZ0();
      TheHistManager->GetHisto("ZZZ420")->Fill(ZZZ420);
      zbegin = (z420+zinibeg) - ZZZ420;
      zfinis = ((z420+zinibeg)+zref1) - ZZZ420;
      // info: center of 1st station is at  0.
      zref =    zref1 + (z420+zinibeg) - ZZZ420;     // zref is z from the vertex of Primary to zfinis
      TheHistManager->GetHisto("XXX420")->Fill(XXX420);
      TheHistManager->GetHisto("YYY420")->Fill(YYY420);
      TheHistManager->GetHisto2("2DXY420")->Fill(XXX420,YYY420);
      TheHistManager->GetHisto("npart420")->Fill(float(npart));
      if(npart !=1)std::cout << "DigFP420Test::warning: NumberOfPrimaryPart != 1--> = " <<npart<<std::endl;
#ifdef ddebugprim1
      std::cout << "zref = " << zref << "(z420+zinibeg) = " << (z420+zinibeg) << "ZZZ420 = " << ZZZ420 << std::endl;
      std::cout << "number of particles for Vertex = " << npart << std::endl;
#endif
      if (npart==0)std::cout << "DigFP420Test: End Of Event ERR: no NumberOfParticle" << std::endl;
      
      // primary vertex (with Hector use it is at z of FP420):
      //	 G4double vx=0.,vy=0.,vz=0.;
      vx = avertex->GetX0();
      vy = avertex->GetY0();
      vz = avertex->GetZ0();
      TheHistManager->GetHisto("VtxX0")->Fill(vx);
      TheHistManager->GetHisto("VtxX")->Fill(vx);
      TheHistManager->GetHisto("VtxY")->Fill(vy);
      TheHistManager->GetHisto("VtxZ")->Fill(vz);
      //  std::cout << "zref = " << zref << "z420 = " << z420 << "ZZZ420 = " << ZZZ420 << std::endl;
      // =============================================================loop over particles of ZZZ420 vertex
      for (int i = 0 ; i<npart; ++i) {
	//loop over particles
#ifdef ddebugprim1
	std::cout << " ----------------   npart  ---------" << std::endl;
#endif
	thePrim=avertex->GetPrimary(i);
	G4ThreeVector   mom  = thePrim->GetMomentum();
	G4int         id   = thePrim->GetTrackID();

	//	G4String       particleType   = thePrim->GetDefinition()->GetParticleName();   //   !!!

	// =====================================
	//  avertex->GetTotalEnergy()    mom.mag()   mom.t()    mom.vect().mag() 
	//    std::cout << "mom.mag() = " << mom.mag() << std::endl;
	////    std::cout << "mom.t() = " << mom.t() << std::endl;
	////    std::cout << "mom.vect().mag() = " << mom.vect().mag() << std::endl;
	////    std::cout << "thePrim->GetTotalEnergy() = " << thePrim->GetTotalEnergy() << std::endl;
	// =====================================
	++icurtrack;
	if(icurtrack==1){
	  phi = mom.phi();
	  if (phi < 0.) phi += twopi;
	  phigrad = phi*180./pi;
	  th     = mom.theta();
	  eta = -log(tan(th/2));
	  ppmom = mom.mag();
	  idmom = id;
	  xi = 1.-mom.mag()/7000000.;
	  
	  bxtrue = tan(th)*cos(phi);
	  bytrue = tan(th)*sin(phi);
	  
	  xref = vx + zref*bxtrue;
	  yref = vy + zref*bytrue;
	  TheHistManager->GetHisto2("2DXY420refLast")->Fill(xref,yref);
	  TheHistManager->GetHisto2("2DXY420refBeg")->Fill( (vx + zbegin*bxtrue) , (vy + zbegin*bytrue) );
	  //	std::cout << " xref = " << xref << " vx = " << vx << " zref = " << zref << " (zref-vz) = " << (zref-vz) << " vz = " << vz << " bxtrue = " << bxtrue << std::endl;
#ifdef ddebugprim1
	  std::cout << "xref = " << xref << "vx = " << vx << "vz = " << vz << "bxtrue = " << bxtrue << std::endl;
	  std::cout << "============================= " << std::endl;
	  std::cout << "DigFP420Test: vx = " << XXX420 << " th=" << th << " phi=" << phi << " xref=" << xref << std::endl;
	  std::cout << " tan(th) = " << tan(th) << " cos(phi)=" << cos(phi) << " bxtrue=" << bxtrue << std::endl;
#endif
	  //  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
#ifdef ddebugprim1
	  std::cout << " lastpo.x()=" << lastpo.x() << std::endl;
	  std::cout << " lastpo.y()=" << lastpo.y() << std::endl;
	  std::cout << " lastpo.z()=" << lastpo.z() << std::endl;
#endif
	  //	  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	 // if(lastpo.z()< zmilimit || (lastpo.z()>zmilimit &&( lastpo.y()>5. || lastpo.y()<-5. || lastpo.x()<-24.7 || lastpo.x()>-4.7)) ) {
	  if(lastpo.z()< zmilimit  ) {
	    varia = 1;    // MI happened
	  }
	  else{
	    varia = 2;   // no MI 
	  } 
	  
	  TheHistManager->GetHisto("PrimaryIDMom")->Fill(float(idmom));
	  TheHistManager->GetHisto("PrimaryMom")->Fill(ppmom);
	  TheHistManager->GetHisto("PrimaryXi0")->Fill(xi);
	  TheHistManager->GetHisto("PrimaryXi")->Fill(xi);
	  TheHistManager->GetHisto("PrimaryXiLog")->Fill(TMath::Log10(xi));
	  TheHistManager->GetHisto("PrimaryEta")->Fill(eta);
	  TheHistManager->GetHisto("xref")->Fill(xref,1.);
	  TheHistManager->GetHisto("yref")->Fill(yref,1.);
	  TheHistManager->GetHisto("PrimaryPhigrad")->Fill(phigrad);
	  TheHistManager->GetHisto("PrimaryTh")->Fill(th*1000.);// mlrad
	}
	// second track:
	else if(icurtrack==2){
	  idmom2 = id;
	  phi2= mom.phi();
	  if (phi2< 0.) phi2 += twopi;
	  phigrad2 = phi2*180./pi;
	  th2     = mom.theta();
	  eta2 = -log(tan(th2/2));
	  ppmom2 = mom.mag();
	  xi2 = 1.-mom.mag()/7000000.;
	  // 2st primary track 
	  bxtrue2= tan(th2)*cos(phi2);
	  bytrue2= tan(th2)*sin(phi2);
	  xref2= vx + zref*bxtrue2;
	  yref2= vy + zref*bytrue2;
#ifdef ddebugprim1
	  std::cout << "DigFP420Test: vx = " <<  XXX420<< " th2=" << th2 << " phi2=" << phi2 << " xref2=" << xref2 << std::endl;
	  std::cout << " tan(th2) = " << tan(th2) << " cos(phi2)=" << cos(phi2) << " bxtrue2=" << bxtrue2 << std::endl;
	  std::cout << " idmom2 = " << idmom2 << " yref2=" << yref2 << " bytrue2=" << bytrue2 << std::endl;
#endif
	  
	  //  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	  //if(  lastpo.z()< zmilimit  || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	  if(  lastpo.z()< zmilimit  ) {
	    varia2= 1;
	  }
	  else{
	    varia2= 2;
	  } 
	  
	  dref12 = abs(xref2 - xref);
	  drefy12 = abs(yref2 - yref);
	  TheHistManager->GetHisto("xref2")->Fill(xref2,1.);
	  TheHistManager->GetHisto("dref12")->Fill(dref12);
	  TheHistManager->GetHisto("drefy12")->Fill(drefy12);
	  TheHistManager->GetHisto("yref2")->Fill(yref2,1.);
	  TheHistManager->GetHisto("thetaX2mrad")->Fill(fabs(bxtrue)*1000.);
	  TheHistManager->GetHisto("PrimaryIDMom2")->Fill(float(idmom2));
	  TheHistManager->GetHisto("PrimaryMom2")->Fill(ppmom2);
	  TheHistManager->GetHisto("PrimaryTh2")->Fill(th2*1000.);// mlrad
	  TheHistManager->GetHisto("PrimaryPhigrad2")->Fill(phigrad2);
	  TheHistManager->GetHisto("PrimaryEta2")->Fill(eta2);
	  TheHistManager->GetHisto("PrimaryXi2")->Fill(xi2);
	  //	TheHistManager->GetHisto("thetaX2mrad")->Fill(fabs(bxtrue2)*1000.);
	}
	else if(icurtrack==3){
	  idmom3 = id;
	  phi3 = mom.phi();
	  if (phi3 < 0.) phi3 += twopi;
	  phigrad3 = phi3*180./pi;
	  th3     = mom.theta();
	  eta3 = -log(tan(th3/2));
	  ppmom3 = mom.mag();
	  xi3 = 1.-mom.mag()/7000000.;
	  // 3rd primary track 
	  bxtrue3= tan(th3)*cos(phi3);
	  bytrue3= tan(th3)*sin(phi3);
	  xref3= vx + zref*bxtrue3;
	  yref3= vy + zref*bytrue3;
	  
	  
	  if(  lastpo.z()< zmilimit  ) {
	    varia3= 1;
	  }
	  else{
	    varia3= 2;
	  } 
	  
	  //                                                                              .
	  TheHistManager->GetHisto("PrimaryIDMom3")->Fill(float(idmom3));
	  TheHistManager->GetHisto("PrimaryMom3")->Fill(ppmom3);
	  TheHistManager->GetHisto("PrimaryTh3")->Fill(th3*1000.);// mlrad
	  TheHistManager->GetHisto("PrimaryPhigrad3")->Fill(phigrad3);
	  TheHistManager->GetHisto("PrimaryEta3")->Fill(eta3);
	  TheHistManager->GetHisto("PrimaryXi3")->Fill(xi3);
	}
	else {
	  std::cout << "DigFP420Test:WARNING i>3" << std::endl; 
	}// if(i
	// =====================================
	
#ifdef ddebugprim1
	std::cout << " i=" << i << "DigFP420Test: at 420m mom = " << mom 
		  << std::endl;
#endif
#ifdef ddebugprim1
	std::cout << " -------------------------------------------------------------" << std::endl;
	std::cout << "DigFP420Test: Vertex vx = " << vx << " vy=" << vy << " vz=" << vz << std::endl;
	std::cout << " Vertex vx = " << vx << " vy=" << vy << "vz=" << vz << std::endl;
	std::cout << " varia = " << varia << " varia2=" << varia2 << " i=" << i << std::endl;
#endif
      }// loop over particles of ZZZ420 vertex  (int i
      
      
      
      //                                                                              .
#ifdef ddebugprim1
      std::cout << " dref12 = " << dref12 << std::endl;
      std::cout << " ================================================================= " << std::endl;
#endif
      
      
  //                                                                              preparations:
  //temporary:
  //  eta = eta0;
  //  xi = xi0;
    //                                                                              .
    //                                                                              .

    }//if(fabs(ZZZ420)
  }// prim.vertex loop end
      TheHistManager->GetHisto("icurtrack")->Fill(float(icurtrack));

//=========================== thePrim != 0 ================================================================================up to end....
//    if (thePrim != 0   && vz < -20.) {

		  
//ask 1 tracks	  	  
		  
//	    std::cout << " ZZZ420 = " << ZZZ420 << " thePrim=" << thePrim << std::endl;
//	    std::cout << " xref = " << xref << " yref=" << yref << std::endl;
//	  if((xref > -25. && xref < -5.) && (yref > -5. && yref < 5.)){
//	    std::cout << " dref12 = " << dref12 << std::endl;
//	  }

//  	     && ((vx > -24.7 && vx < -4.7) && (vy > -5. && vy < 5.))  


//	if (ZZZ420 != -999999.

//	if ( ZZZ420 == -999999.
    // events in detector acceptance :    
  TheHistManager->GetHisto("numofpart0")->Fill(float(numofpart));
#ifdef ddebugprim1
      std::cout << " numofpart = " << numofpart << std::endl;
      std::cout << " ZZZ420 = " << ZZZ420 << std::endl;
#endif
  if ( thePrim != 0 && ZZZ420 != -999999.
       //	     &&	varia == 2  

      // && ((xref > -24.7 && xref < -4.7) && (yref > -5. && yref < 5.)||  
       //	       (xref2 > -24.7 && xref2 < -4.7) && (yref2 > -5. && yref2 < 5.)|| 
       //	       (xref3 > -24.7 && xref3 < -4.7) && (yref3 > -5. && yref3 < 5.))  
       // take into account  rescattering in flat pocket part: go from 4.7 to 4.5   (thickness 0.2mm  )  
       && ((xref > -24.7 && xref < -4.5) && (yref > -5. && yref < 5.)||  
	       (xref2 > -24.7 && xref2 < -4.5) && (yref2 > -5. && yref2 < 5.)|| 
	       (xref3 > -24.7 && xref3 < -4.5) && (yref3 > -5. && yref3 < 5.))  
       
       ) {
    // events in detector acceptance :    


#ifdef ddebugprim1
      std::cout << " xref = " << xref << std::endl;
      std::cout << " yref = " << yref << std::endl;
#endif
	 TheHistManager->GetHisto("xrefAcc")->Fill(xref,1.);
	 TheHistManager->GetHisto("yrefAcc")->Fill(yref,1.);


    
//
//	     &&	varia == 2  
//	     && ((xref > -32. && xref < -12.) && (yref > -5. && yref < 5.))  
//	     &&	varia == 2  
//	     && ( fabs(bxtrue)*1000. > 0.1  && fabs(bxtrue)*1000.<0.4 )
		  
// ask 2 tracks		  
/*
	if ( thePrim != 0  && ZZZ420 != -999999.
	     && ((xref  > -25. && xref  < -5.) && (yref  > -5. && yref  < 5.))  
	     && ((xref2 > -25. && xref2 < -5.) && (yref2 > -5. && yref2 < 5.))  
	     && dref12 > 1.0 && drefy12 > 1.0       
	     ) {
*/	  
	  
	  
    // ==========================================================================
    
    // hit map for FP420
    // ==================================
    
    map<int,float,less<int> > themap;
    map<int,float,less<int> > themap1;
    
    map<int,float,less<int> > themapxystrip;
    map<int,float,less<int> > themapxy;
    map<int,float,less<int> > themapxystripW;
    map<int,float,less<int> > themapxyW;
    map<int,float,less<int> > themapz;
    // access to the G4 hit collections:  -----> this work OK:
    
    G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
    
    if (verbosity > 0) {
      std::cout << "DigFP420Test:  accessed all HC" << std::endl;;
    }
    int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("FP420SI");
    
    FP420G4HitCollection* theCAFI = (FP420G4HitCollection*) allHC->GetHC(CAFIid);
    if (verbosity > 0) {
      //std::cout << "FP420Test: theCAFI = " << theCAFI << std::endl;
      std::cout << "DigFP420Test: theCAFI->entries = " << theCAFI->entries() << std::endl;
    }
    TheHistManager->GetHisto("NHitsAll")->Fill(theCAFI->entries());
    
   // TheHistManager->GetHisto2("2DSecVsR")->Fill(float(numofpart),lastpo.perp());
   // TheHistManager->GetHisto2("2DSecVsZ")->Fill(float(numofpart),lastpo.z());
   // TheHistManager->GetHisto2("2DSecVsHits")->Fill(float(numofpart),theCAFI->entries());
   // TheHistManager->GetHisto2("2DHitsVsR")->Fill(theCAFI->entries(),lastpo.perp());
   // TheHistManager->GetHisto2("2DHitsVsZ")->Fill(theCAFI->entries(),lastpo.z());

    //    if(numofpart ==  0) {   
     // if(varia == 1){
//	TheHistManager->GetHisto("NumberHitsOnlyMI")->Fill(theCAFI->entries());
     // }
     // else {
//	TheHistManager->GetHisto("NumberHitsNoMI")->Fill(theCAFI->entries());
     // }
      //   }
    int mhits1=0, mhits2=0, mhits3=0, mhitstot=0;
    for (int j=0; j<theCAFI->entries(); j++) {
      FP420G4Hit* aHit = (*theCAFI)[j];
      unsigned int unitID = aHit->getUnitID();
      int det, zside, sector, zmodule;
      FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
      if ( abs(aHit->getTof()) < 100. && aHit->getEnergyLoss() >0) {
	if(sector==1) ++mhits1;
	if(sector==2) ++mhits2;
	if(sector==3) ++mhits3;
	++mhitstot;
      }
    }
    TheHistManager->GetHisto("NumofHitsSec1")->Fill(float(mhits1));
    TheHistManager->GetHisto("NumofHitsSec2")->Fill(float(mhits2));
    TheHistManager->GetHisto("NumofHitsSec3")->Fill(float(mhits3));
    
    int metmi=-1;
    // NoMIevents in detector acceptance :    
    // if(numofpart == 0){
    if(mhitstot<31 && (mhits1>2 && mhits1<11) && (mhits2>2 && mhits2<11) && (mhits3>2 && mhits3<11)){
      //      if(theCAFI->entries()<36 && (mhits1>2 && mhits1<11) && (mhits2>2 && mhits2<11) && (mhits3>2 && mhits3<16)){
      metmi=0;
      TheHistManager->GetHisto("NumofpartNoMI")->Fill(float(numofpart));
      TheHistManager->GetHisto("PrimaryLastpoX")->Fill(lastpo.x());
      TheHistManager->GetHisto("PrimaryLastpoY")->Fill(lastpo.y());
      TheHistManager->GetHisto("PrimaryLastpoZ")->Fill(lastpo.z());
      TheHistManager->GetHisto("NumberHitsNoMI")->Fill(theCAFI->entries());
      TheHistManager->GetHisto("xrefNoMI")->Fill(xref,1.);
      TheHistManager->GetHisto("yrefNoMI")->Fill(yref,1.);
    }
    // MI events (in detector acceptance )   
    //  if(numofpart != 0){
    else {
      metmi=1;
      TheHistManager->GetHisto("NumofpartOnlyMI")->Fill(float(numofpart));
      TheHistManager->GetHisto("XLastpoNumofpart")->Fill(lastpo.x());
      TheHistManager->GetHisto("YLastpoNumofpart")->Fill(lastpo.y());
      TheHistManager->GetHisto("PrimaryLastpoZ0")->Fill(lastpo.z());
      TheHistManager->GetHisto("NumberHitsOnlyMI")->Fill(theCAFI->entries());
      TheHistManager->GetHisto("xrefMI")->Fill(xref,1.);
      TheHistManager->GetHisto("yrefMI")->Fill(yref,1.);
    }
    
      int metmi2=-1;
      if(mhitstot<61 && (mhits1>2 && mhits1<21) && (mhits2>2 && mhits2<21) && (mhits3>2 && mhits3<21)){
	metmi2=0;
	TheHistManager->GetHisto("xref2NoMI")->Fill(xref,1.);
	TheHistManager->GetHisto("xref2NoMI")->Fill(xref2,1.);
      }
      else {
	metmi2=1;
	TheHistManager->GetHisto("xref2MI")->Fill(xref,1.);
	TheHistManager->GetHisto("xref2MI")->Fill(xref2,1.);
      }
      
    
       //   varia = 0;
       // one particle case:
      // if( varia == 2  && numofpart ==  0  && theCAFI->entries() < 31 ) {

       // ask one hit at least
       if( theCAFI->entries() !=0 ) {


	 //   if( varia == 2  && numofpart ==  0) {
	 //    if( theCAFI->entries() < 49 && numofpart ==  0) {
	 // if( numofpart ==  0) {
	 //varia      
	 double  totallosenergy= 0.;
	 int AATest[80];
      for (int j=0; j<80; j++) {
	AATest[j]=0; }
      // loop over hits      
      for (int j=0; j<theCAFI->entries(); j++) {
	FP420G4Hit* aHit = (*theCAFI)[j];
	
	Hep3Vector hitEntryLocalPoint = aHit->getEntryLocalP();
	Hep3Vector hitExitLocalPoint = aHit->getExitLocalP();
	Hep3Vector hitPoint = aHit->getEntry();
	G4ThreeVector middle = (hitExitLocalPoint+hitEntryLocalPoint)/2.;
	G4ThreeVector mid    = (hitExitLocalPoint-hitEntryLocalPoint)/2.;
	unsigned int unitID = aHit->getUnitID();
	double  losenergy = aHit->getEnergyLoss();
	TheHistManager->GetHisto("EntryX")->Fill(hitPoint.x());
	TheHistManager->GetHisto("EntryY")->Fill(hitPoint.y());
	TheHistManager->GetHisto("midZ")->Fill(mid.z());
	/*
	  int trackIDhit  = aHit->getTrackID();
	  double  elmenergy =  aHit->getEM();
	  double  hadrenergy =  aHit->getHadr();
	  double incidentEnergyHit  = aHit->getIncidentEnergy();
	  double   timeslice = aHit->getTimeSlice();     
	  int     timesliceID = aHit->getTimeSliceID();     
	  double  depenergy = aHit->getEnergyDeposit();
	  float   pabs = aHit->getPabs();
	  float   tof = aHit->getTof();
	  int   particletype = aHit->getParticleType();
	  float thetaEntry = aHit->getThetaAtEntry();   
	  float phiEntry = aHit->getPhiAtEntry();
	  float xEntry = aHit->getX();
	  float yEntry = aHit->getY();
	  float zEntry = aHit->getZ();
	  int  parentID = aHit->getParentId();
	  float vxget = aHit->getVx();
	  float vyget = aHit->getVy();
	  float vzget = aHit->getVz();
	*/
	
#ifdef mmydebug
	std::cout << "======================Hit Collection" << std::endl;
	std::cout << "lastpo.x() = " << lastpo.x() << std::endl;
	std::cout << "lastpo.y() = " << lastpo.y() << std::endl;
	std::cout << "lastpo.z() = " << lastpo.z() << std::endl;
	std::cout << "hitPoint = " << hitPoint << std::endl;
	std::cout << "hitEntryLocalPoint = " << hitEntryLocalPoint << std::endl;
	std::cout << "hitExitLocalPoint = " << hitExitLocalPoint << std::endl;
	std::cout << "elmenergy = " << elmenergy << "hadrenergy = " << hadrenergy << std::endl;
	std::cout << "incidentEnergyHit = " << incidentEnergyHit << "trackIDhit = " << trackIDhit << std::endl;
	std::cout << "unitID=" << unitID <<std::endl;
	std::cout << "timeslice = " << timeslice << "timesliceID = " << timesliceID << std::endl;
	std::cout << "depenergy = " << depenergy << "pabs = " << pabs  << std::endl;
	std::cout << "tof = " << tof << "losenergy = " << losenergy << std::endl;
	std::cout << "particletype = " << particletype << "thetaEntry = " << thetaEntry << std::endl;
	std::cout << "phiEntry = " << phiEntry << "xEntry = " << xEntry  << std::endl;
	std::cout << "yEntry = " << yEntry << "zEntry = " << zEntry << std::endl;
	std::cout << "parentID = " << parentID << "vxget = " << vxget << std::endl;
	std::cout << "vyget = " << vyget << "vzget = " << vzget << std::endl;
#endif
    TheHistManager->GetHisto("HitLosenergy")->Fill(losenergy);
    TheHistManager->GetHisto("Hithadrenergy")->Fill(aHit->getHadr());
    TheHistManager->GetHisto("HitIncidentEnergy")->Fill(aHit->getIncidentEnergy());
    TheHistManager->GetHisto("HitTimeSlice")->Fill(aHit->getTimeSlice());
    TheHistManager->GetHisto("HitEnergyDeposit")->Fill(aHit->getEnergyDeposit());
    TheHistManager->GetHisto("HitPabs")->Fill(aHit->getPabs());
    TheHistManager->GetHisto("HitTof")->Fill(aHit->getTof());
    TheHistManager->GetHisto("HitParticleType")->Fill(aHit->getParticleType());
    TheHistManager->GetHisto("HitThetaAtEntry")->Fill(aHit->getThetaAtEntry());
    TheHistManager->GetHisto("HitPhiAtEntry")->Fill(aHit->getPhiAtEntry());
    TheHistManager->GetHisto("HitX")->Fill(aHit->getX());
    TheHistManager->GetHisto("HitY")->Fill(aHit->getY());
    TheHistManager->GetHisto("HitZ")->Fill(aHit->getZ());
    TheHistManager->GetHisto("HitParentId")->Fill(aHit->getParentId());
    TheHistManager->GetHisto("HitVx")->Fill(aHit->getVx());
    TheHistManager->GetHisto("HitVy")->Fill(aHit->getVy());
    TheHistManager->GetHisto("HitVz")->Fill(aHit->getVz());
    //   if(theCAFI->entries()>80) {
    if ( abs(aHit->getTof()) < 100. && losenergy>0) {
      if (j==0 )	TheHistManager->GetHisto("NumberHitsFinal")->Fill(theCAFI->entries());

      if (metmi==0 ) TheHistManager->GetHisto("HitIncidentEnergyNoMI")->Fill(aHit->getIncidentEnergy());
      if (metmi==1 ) TheHistManager->GetHisto("HitIncidentEnergyMI")->Fill(aHit->getIncidentEnergy());

      TheHistManager->GetHisto("HitLosenergyH")->Fill(losenergy);
      TheHistManager->GetHisto("HithadrenergyH")->Fill(aHit->getHadr());
      TheHistManager->GetHisto("HitIncidentEnergyH")->Fill(aHit->getIncidentEnergy());
      TheHistManager->GetHisto("HitTimeSliceH")->Fill(aHit->getTimeSlice());
      TheHistManager->GetHisto("HitEnergyDepositH")->Fill(aHit->getEnergyDeposit());
      TheHistManager->GetHisto("HitPabsH")->Fill(aHit->getPabs());
      TheHistManager->GetHisto("HitTofH")->Fill(aHit->getTof());
      TheHistManager->GetHisto("HitParticleTypeH")->Fill(aHit->getParticleType());
      TheHistManager->GetHisto("HitThetaAtEntryH")->Fill(aHit->getThetaAtEntry());
      TheHistManager->GetHisto("HitPhiAtEntryH")->Fill(aHit->getPhiAtEntry());
      TheHistManager->GetHisto("HitXH")->Fill(aHit->getX());
      TheHistManager->GetHisto("HitYH")->Fill(aHit->getY());
      TheHistManager->GetHisto("HitZH")->Fill(aHit->getZ());
      TheHistManager->GetHisto("HitParentIdH")->Fill(aHit->getParentId());
      TheHistManager->GetHisto("HitVxH")->Fill(aHit->getVx());
      TheHistManager->GetHisto("HitVyH")->Fill(aHit->getVy());
      TheHistManager->GetHisto("HitVzH")->Fill(aHit->getVz());
      TheHistManager->GetHisto("EntryXH")->Fill(hitPoint.x());
      TheHistManager->GetHisto("EntryYH")->Fill(hitPoint.y());
      TheHistManager->GetHisto("midZH")->Fill(mid.z());
      //  } //   if ( abs(aHit->getTof()) < 100. && losenergy>0)
    
    
    //double th_hit    = hitPoint.theta();
    //double eta_hit = -log(tan(th_hit/2));
    double phi_hit   = hitPoint.phi();
    if (phi_hit < 0.) phi_hit += twopi;
    double   zz=-999999.;
    zz    = hitPoint.z();
    if (verbosity > 2) {
      std::cout << "DigFP420Test:zHits = " << zz << std::endl;
    }
    themap[unitID] += losenergy;
    totallosenergy += losenergy;
    
    int det, zside, sector, zmodule;
    FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
    
    //////////////                                                             //////////////
    //test of # hits per every iitest:
    // iitest   is a continues numbering of FP420
    unsigned int iitest = 2*(pn0-1)*(sector - 1)+2*(zmodule - 1)+zside;
    ++AATest[iitest-1];
    //////////////                                                             //////////////
    if(iitest == 1){
      TheHistManager->GetHisto("EntryZ1")->Fill(hitPoint.z()-419900.);
    }
    else if(iitest == 2){
      TheHistManager->GetHisto("EntryZ2")->Fill(hitPoint.z()-419900.);
    }
    else if(iitest == 3){
      TheHistManager->GetHisto("EntryZ3")->Fill(hitPoint.z()-419900.);
    }
    else if(iitest == 4){
      TheHistManager->GetHisto("EntryZ4")->Fill(hitPoint.z()-419900.);
    }
    
    // zside=1,2 ; zmodule=1,10 ; sector=1,3
    if(zside==0||sector==0||zmodule==0){
      std::cout << "DigFP420Test:ERROR: zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << " det = " << det  << std::endl;
    }
    
    double kplane = -(pn0-1)/2-0.5+(zmodule-1); 
    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg + z420 +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    
    if(zside==1){
      zcurrent += (ZGapLDet+ZSiDetL/2);
    }
    if(zside==2){
      zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
    }     
    
    //=======================================
    // SimHit position in Local reference frame - middle :
    
    //
    if (verbosity > 2) {
      std::cout << "DigFP420Test:check " << std::endl;
      std::cout << " zside = " <<zside<< " sector = " <<sector<< " zmodule = " << zmodule<< std::endl;
      std::cout << " hitPoint.z()+ mid.z() = " <<  double (hitPoint.z()+ mid.z()-z420) << std::endl;
      std::cout << " zcurrent = " << double (zcurrent-z420) << " ZGapLDet = " << ZGapLDet << std::endl;
      std::cout << " diff = " << double (hitPoint.z()+ mid.z()- zcurrent) << std::endl;
      std::cout << " zinibeg = " <<zinibeg<< " kplane*ZSiStep = " <<kplane*ZSiStep<< "  zcurrentBefore= " << zinibeg + z420 +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat-z420<< std::endl;
    }
    //=======================================
    //   themapz[unitID]  = hitPoint.z()+ mid.z(); // this line just for studies
    themapz[unitID]  = zcurrent;// finally must be this line !!!
    
    //=======================================
    
    themapxystrip[unitID] = -1.;// charge in strip coord 
    float numStrips,pitch;
    themapxystripW[unitID] = -1.;// charge in stripW coord 
    float numStripsW,pitchW;
    //=======================================
    // Y global
    if(xytype==1) {
      //UserNtuples->fillg24(losenergy,1.);
      if(losenergy > 0.00003) {
	themap1[unitID] += 1.;
      }
      // E field (and p+,n+sides) along X,  but define Y coordinate -->  number of strips of the same side(say p+):200*pitchX=20mm
      numStrips = numStripsY;
      pitch=pitchY;
      themapxystrip[unitID] = 0.5*(numStrips-1) + middle.x()/pitch ;// charge in strip coord 
      themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.x();//hit coor. in l.r.f starting at bot edge of plate
      
      numStripsW = numStripsYW;
      pitchW=pitchYW;
      themapxystripW[unitID] = 0.5*(numStripsW-1) + middle.y()/pitchW ;// charge in strip coord 
      themapxyW[unitID]  = (numStripsW-1)*pitchW/2. + middle.y();//hit coor. in l.r.f starting at bot edge of plate
    }
    //X
    if(xytype==2){
      if(losenergy > 0.00003) {
	themap1[unitID] += 1.;
      }
      numStrips = numStripsX;
      pitch=pitchX;
      themapxystrip[unitID] = 0.5*(numStrips-1) + middle.y()/pitch ;// charge in strip coord 
      themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.y(); //hit coor. in l.rf starting at left edge of plate
      
      numStripsW = numStripsXW;
      pitchW=pitchXW;
      themapxystripW[unitID] = 0.5*(numStripsW-1) + middle.x()/pitchW ;// charge in strip coord 
      themapxyW[unitID]  = (numStripsW-1)*pitchW/2. + middle.x(); //hit coor. in l.rf starting at left edge of plate
      
      
    }    //if
    
    
    // MIonly or noMIonly ENDED
    //    }
    
    //     !!!!!!!!!!!!!
    //     !!!!!!!!!!!!!
    //     !!!!!!!!!!!!!
      } //   if ( abs(aHit->getTof()) < 100. && losenergy>0)
      
    }  // for loop on all hits ENDED  ENDED  ENDED  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
      //     !!!!!!!!!!!!!
      //     !!!!!!!!!!!!!
      //     !!!!!!!!!!!!!
      
      for (int j=0; j<80 && AATest[j]!=0. ; j++) {
	TheHistManager->GetHisto("ATest")->Fill(AATest[j]);
      }
      //DIGI	
      // DIGI and HIt collections have dirrerent # hits since the cuts in FP420DigiMain were applied (like Tof, Eloss...)      
      
      //=================================================================================
      //
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //                                                                 DIGI:                                                   .
      //
      //=================================================================================
                                                                                                 


          
//====================================================================================================== number of hits

// Digi validation:
      if(totallosenergy == 0.0) {
      } else{
	// =totallosenergy==================================================================================   produce Digi start
	//    produce();
	
	theDigitizerFP420->produce(theCAFI,output);
	
	if (verbosity > 2) {
	  std::cout <<" ===== DigFP420Test:: access to DigiCollectionFP420 " << std::endl;
	}
	//    check of access to the strip collection
	// =======================================================================================check of access to strip collection
	if (verbosity > 2) {
	  std::cout << "DigFP420Test:  start of access to the collector" << std::endl;
	}
	for (int sector=1; sector<sn0; sector++) {
	  for (int zmodule=1; zmodule<pn0; zmodule++) {
	    for (int zside=1; zside<3; zside++) {
	      //int det= 1;
	      //int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	      
	      // intindex is a continues numbering of FP420
	      int sScale = 2*(pn0-1);
	      int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	      // int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
	      
	      if (verbosity > 2) {
		std::cout <<" ===== DigFP420Test:: sector= " << sector  
			  << "zmodule= "  << zmodule  
			  << "zside= "  << zside  
			  << "iu= "  << iu  
			  << std::endl;
	      }
	      std::vector<HDigiFP420> collector;
	      collector.clear();
	      DigiCollectionFP420::Range outputRange;
	      outputRange = output.get(iu);
	      // fill output in collector vector (for may be sorting? or other checks)
	      DigiCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	      DigiCollectionFP420::ContainerIterator sort_end = outputRange.second;
	      
	      for ( ;sort_begin != sort_end; ++sort_begin ) {
		collector.push_back(*sort_begin);
	      } // for
	      
	      
		// map to store pixel Amplitudesin the x and in the y directions
		// map<int, float, less<int> > AmplitudeX,AmplitudeXW; 
		// map<int, float, less<int> > AmplitudeY,AmplitudeYW; 
	      double AmplitudeX[401],AmplitudeXW[26]; 
	      double AmplitudeY[201],AmplitudeYW[52]; 
	      for (int im=0;  im<numStripsY;  ++im) {
		AmplitudeY[im] = 0.;
	      }
	      for (int im=0;  im<numStripsYW;  ++im) {
		AmplitudeYW[im] = 0.;
	      }
	      
	      for (int im=0;  im<numStripsX;  ++im) {
		AmplitudeX[im] = 0.;
	      }
	      for (int im=0;  im<numStripsXW;  ++im) {
		AmplitudeXW[im] = 0.;
	      }
	      
	      
	      vector<HDigiFP420>::const_iterator simHitIter = collector.begin();
	      vector<HDigiFP420>::const_iterator simHitIterEnd = collector.end();
	      for (;simHitIter != simHitIterEnd; ++simHitIter) {
		const HDigiFP420 istrip = *simHitIter;
		// Y:
		if(xytype==1){
		  int iy= istrip.channel()/numStripsY;
		  int ix= istrip.channel() - iy*numStripsY;
		  AmplitudeY[ix] = + istrip.adc();
		  AmplitudeYW[iy] = + istrip.adc();
		  //  float pitch=pitchY;
		  double moduleThickness = ZSiDetL; 
		  //float sigmanoise =  ENC*ldriftX/Thick300/ElectronPerADC;
		  float sigmanoise =  ENC*moduleThickness/Thick300/ElectronPerADC;
		  TheHistManager->GetHisto("DigiYstrip")->Fill(ix);
		  TheHistManager->GetHisto("DigiYWstrip")->Fill(iy);
		  TheHistManager->GetHisto2("2DigiYYW")->Fill(iy,ix);
		  TheHistManager->GetHisto2("2DigiYYWAmplitude")->Fill(iy,ix,istrip.adc());
		  TheHistManager->GetHisto("DigiYstripAdc")->Fill(istrip.adc());
		  TheHistManager->GetHisto("DigiYstripAdcSigma")->Fill(istrip.adc()/sigmanoise);
		  TheHistManager->GetHisto("DigiAmplitudeY")->Fill(ix,istrip.adc());
		  TheHistManager->GetHisto("DigiAmplitudeYW")->Fill(iy,istrip.adc());
		}
		// X:
		else if(xytype==2){
		  int iy= istrip.channel()/numStripsX;
		  int ix= istrip.channel() - iy*numStripsX;
		  AmplitudeX[ix] = + istrip.adc();
		  AmplitudeXW[iy] = + istrip.adc();
		  //  float pitch=pitchX;
		  double moduleThickness = ZSiDetR; 
		  //float sigmanoise =  ENC*ldriftY/Thick300/ElectronPerADC;
		  float sigmanoise =  ENC*moduleThickness/Thick300/ElectronPerADC;
		  TheHistManager->GetHisto("DigiXstrip")->Fill(ix);
		  TheHistManager->GetHisto("DigiXWstrip")->Fill(iy);
		  TheHistManager->GetHisto2("2DigiXXW")->Fill(iy,ix);
		  TheHistManager->GetHisto2("2DigiXXWAmplitude")->Fill(iy,ix,istrip.adc());
		  TheHistManager->GetHisto("DigiXstripAdc")->Fill(istrip.adc());
		  TheHistManager->GetHisto("DigiXstripAdcSigma")->Fill(istrip.adc()/sigmanoise);
		  TheHistManager->GetHisto("DigiXWstripAdcSigma")->Fill(istrip.adc()/sigmanoise);
		  TheHistManager->GetHisto("DigiAmplitudeX")->Fill(ix,istrip.adc());
		  TheHistManager->GetHisto("DigiAmplitudeXW")->Fill(iy,istrip.adc());
		  if(sector==1){
		    TheHistManager->GetHisto("DigiXstripAdcSigma1")->Fill(istrip.adc()/sigmanoise);
		  }
		  else if(sector==2){
		    TheHistManager->GetHisto("DigiXstripAdcSigma2")->Fill(istrip.adc()/sigmanoise);
		  }
		  else if(sector==3){
		    TheHistManager->GetHisto("DigiXstripAdcSigma3")->Fill(istrip.adc()/sigmanoise);
		  }
		}
		//==================================
	      }// forsimHitIter
	      
	      for (int im=0;  im<numStripsY;  ++im) {
		TheHistManager->GetHisto("AmplitudeY")->Fill(AmplitudeY[im]);
	      }
	      for (int im=0;  im<numStripsYW;  ++im) {
		TheHistManager->GetHisto("AmplitudeYW")->Fill(AmplitudeYW[im]);
	      }
	      
	      for (int im=0;  im<numStripsX;  ++im) {
		TheHistManager->GetHisto("AmplitudeX")->Fill(AmplitudeX[im]);
	      }
	      for (int im=0;  im<numStripsXW;  ++im) {
		TheHistManager->GetHisto("AmplitudeXW")->Fill(AmplitudeXW[im]);
	      }
	      
	      //==================================
	    }   // for
	  }   // for
	}   // for
	// =================================================================================DIGI end
	
       
       
      } // if(totallosenergy 
      //====================================================================================================== number of hits
    }   // MI or no MI or all  - end
    else{
      //#ifdef mydebug10
      std::cout << "Else: varia: MI or no MI or all " << std::endl;
      //#endif
    }
    
	}                                                // primary end
	else{
	  //#ifdef mydebug10
	  std::cout << "Else: primary  " << std::endl;
	  //#endif
	}
	//=========================== thePrim != 0  end   ================================================================================
	
	
}
// ==========================================================================

