///////////////////////////////////////////////////////////////////////////////
// File: RecFP420Test
// Date: 02.2007
// Description: RecFP420Test for FP420
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

//#include "SimRomanPot/SimFP420/interface/ClusterFP420.h"
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
#include <gsl/gsl_fit.h>
//#include <gsl/gsl_cdf.h>


//================================================================
#include <cassert>

using namespace edm;
using namespace std;
///////////////////////////////////////////////////////////////////////////////

//#define mydigidebug10
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
   
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   std::cout << "DigFP420Test constructor :: Initialized as observer"<< std::endl;
  }
	

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

  z1 = -150. + (118.4+10.)/2 + z420; // 10. -arbitrary //z1 - right after 1st Station
  z2 = z1+zD2;                       //z2 - right after 2nd Station
  z3 = z1+zD3;
  z4 = z1+2*zD3;
  //==================================
  if (verbosity > 0) {
   std::cout<<"============================================================================"<<std::endl;
   // std::cout << "DigFP420Test constructor :: Initialized as observer zUnit=" << zUnit << std::endl;
   std::cout << "DigFP420Test constructor :: Initialized as observer zD2=" << zD2 << std::endl;
   std::cout << " zD3=" << zD3 << std::endl;
   std::cout << " z1=" << z1 << " z2=" << z2 << " z3=" << z3 << " z4=" << z4 << std::endl;
  }
  //==================================

 fp420eventntuple = new TNtuple("NTfp420event","NTfp420event","evt");
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
  UseHalfPitchShiftInX_= true;
  //UseHalfPitchShiftInX_= false;
  
  UseHalfPitchShiftInY_= true;
  //UseHalfPitchShiftInY_= false;
  
  //-------------------------------------------------
  //UseThirdPitchShiftInY_ = true;
  UseThirdPitchShiftInY_ = false;
  
  //UseThirdPitchShiftInX_ = true;
  UseThirdPitchShiftInX_ = false;
  
  //-------------------------------------------------
  //UseForthPitchShiftInY_ = true;
  UseForthPitchShiftInY_ = false;
  
  //UseForthPitchShiftInX_ = true;
  UseForthPitchShiftInX_ = false;
  
  //-------------------------------------------------
	ldriftX= 0.050;
	ldriftY= 0.050;// was 0.040
	
	pitchX= 0.050;
	pitchY= 0.050;// was 0.040
	
	numStripsY = 201;        // Y plate number of strips:200*0.050=10mm (zside=1)
	numStripsX = 401;        // X plate number of strips:400*0.050=20mm (zside=2)
	numStripsYW = 51;        // Y plate number of W strips:50 *0.400=20mm (zside=1) - W have ortogonal projection
	numStripsXW = 26;        // X plate number of W strips:25 *0.400=10mm (zside=2) - W have ortogonal projection
	

	//  BoxYshft = [gap]+[dYcopper]+[dYsteel] = +12.3 + 0.05 + 0.15 = 12.5  ;  dYGap   =      0.2 mm
	//  dXXconst = 12.7+0.05;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7+0.05
        //	dXXconst = 12.7;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7

	//	dXXconst = 4.7;                     // gap = 4.3 instead 12.3
	//	dYYconst = 5.;// XSiDet/2.

	//ENC = 1800;
	//ENC = 3000;
	ENC = 2160;
	//	ElectronPerADC =300;
	Thick300 = 0.300;
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

  // Initialization:

	theFP420NumberingScheme = new FP420NumberingScheme();
//
}



DigFP420Test::~DigFP420Test() {
  //  delete UserNtuples;
  delete theFP420NumberingScheme;
  delete theDigitizerFP420;

  TFile fp420OutputFile("newntfp420.root","RECREATE");
  std::cout << "DigFP420Test output root file has been created";
  fp420eventntuple->Write();
  std::cout << ", written";
  fp420OutputFile.Close();
  std::cout << ", closed";
  delete fp420eventntuple;
  std::cout << ", and deleted" << std::endl;

        //------->while end

        // Write histograms to file
        TheHistManager->WriteToFile(fOutputFile,fRecreateFile);

  if (verbosity > 0) {
    std::cout << std::endl << "DigFP420Test Destructor  -------->  End of DigFP420Test : "
      << std::cout << std::endl; 
  }

  std::cout<<"DigFP420Test: End of process"<<std::endl;

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

    HistInit("MSC2X",    "MSC2X",   100, -0.015,0.015);
    HistInit("MSC3X",    "MSC3X",   100, -0.025,0.025);
    HistInit("MSC2Y",    "MSC2Y",   100, -0.015,0.015);
    HistInit("MSC3Y",    "MSC3Y",   100, -0.025,0.025);


    HistInit("NumofpartNoMI", "Numofpart No MI",   100,   0.,200. );
    HistInit("NumberHitsNoMI", "Number Hits No MI",   100,   0.,200. );
    HistInit("NumofpartOnlyMI", "Numofpart Only MI",   100,   0.,200. );
    HistInit("NumberHitsOnlyMI", "Number Hits Only MI",   100,   0.,200. );
    HistInit("NHitsAll", "N Hits All",   100,   0.,100. );

    HistInit("PrimaryXi",   "Primary Xi",        100,   0.001, 1.);
    HistInit("PrimaryXiLog","Primary Xi Log",    50,  -3.,     0.);
    HistInit("PrimaryEta",  "Primary Eta",       50,   8.,    13. );
    HistInit("PrimaryXiTr",   "Primary XiTr",    100,   0.001, 1.);
    HistInit("PrimaryXiTrLog","Primary XiTr Log",50, -3.,     0.);
    HistInit("PrimaryEtaTr",  "Primary EtaTr",   50,   8.,    13. );

    HistInit("PrimaryPhigrad", "Primary Phigrad",   100,   0.,360. );
    HistInit("PrimaryTh", "Primary Th",   100,   0.,-0.5 );
    HistInit("PrimaryLastpoZ", "Primary Lastpo Z",   100, -90.,12000. );
    HistInit("PrimaryLastpoX", "Primary Lastpo X Z<z4",   100, -30., 30. );
    HistInit("PrimaryLastpoY", "Primary Lastpo Y Z<z4",   100, -30., 30. );
    HistInit("XLastpoNumofpart", "Primary Lastpo X n>10",   100, -30., 30. );
    HistInit("YLastpoNumofpart", "Primary Lastpo Y n>10",   100, -30., 30. );
    HistInit("VtxX", "Vtx X",   100, -40.,+10. );
    HistInit("VtxY", "Vtx Y",   100, -4., 4. );
    HistInit("VtxZ", "Vtx Z",   100, 400000., 420000. );
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
    HistInit("DigiXstripAdcSigma1",  "Digi Xstrip Adc in SigmaNoise1",      mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma2",  "Digi Xstrip Adc in SigmaNoise2",      mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma3",  "Digi Xstrip Adc in SigmaNoise3",      mx,   0.,xm  );
    HistInit("DigiXstripAdcSigma4",  "Digi Xstrip Adc in SigmaNoise4",      mx,   0.,xm  );

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
    HistInit("DigiYstripAdcSigma1",  "Digi Ystrip Adc in SigmaNoise1",      my,   0.,ym );
    HistInit("DigiYstripAdcSigma2",  "Digi Ystrip Adc in SigmaNoise2",      my,   0.,ym );
    HistInit("DigiYstripAdcSigma3",  "Digi Ystrip Adc in SigmaNoise3",      my,   0.,ym );
    HistInit("DigiYstripAdcSigma4",  "Digi Ystrip Adc in SigmaNoise4",      my,   0.,ym );

	 // Clusters:
    HistInit("NumOfClusters", "Number Of Clusters",   10,   0.,   10.);
    HistInit("NumOfClusMI", "Number Of Clusters MI",   10,   0.,   10.);
    HistInit("NumOfClusNoMI", "Number Of Clusters NoMI",   10,   0.,   10.);
    HistInit("NclPerPlane", "NclPerPlane",   10,   0.,   10.);
    // vs:
    HistInit("ZSimHit","ZSimHit",50, 0.,50.);
    HistInit("ZSimHitNumbCl","ZSimHit NumbCl",50, 0.,50.);
    //HistInit("ZSimHit","ZSimHit",100, 0.,8000.);
    //HistInit("ZSimHitNumbCl","ZSimHit NumbCl",100, 0.,8000.);

    // 2D:
//    HistInit("2DZSimHitNumbCl","2D ZSimHit NumbCl",100, -200.,8000.,10, 0.,10.);
    HistInit("2DZSimHitNumbCl","2D ZSimHit NumbCl",100, 419000.,429000.,10, 0.,10.);
    // X:
    HistInit("clnum1X", "Number Of Clusters in X 1Station",    11,   0.,   11.);
    HistInit("clnum2X", "Number Of Clusters in X 2Station",    11,   0.,   11.);
    HistInit("clnum3X", "Number Of Clusters in X 3Station",    11,   0.,   11.);
    HistInit("clnum4X", "Number Of Clusters in X 4Station",    11,   0.,   11.);
    HistInit("clnum0X", "Number Of Clusters in X all together",41,   0.,   41.);
    HistInit("clnum1Xinside", "Number Of Clusters in X inside1",41,   0.,   41.);
    HistInit("clnum2Xinside", "Number Of Clusters in X inside2",41,   0.,   41.);
    HistInit("Xstrip_deltaxx", "Xstrip Dx",                        80,   -0.2,   0.2);
    HistInit("Xstrip_deltaxx_clsize1", "Xstrip deltaxx clsize=1",  80,   -0.2,   0.2);
    HistInit("Xstrip_deltaxx_clsize2", "Xstrip deltaxx clsize=2",  80,   -0.2,   0.2);
    HistInit("Xstrip_deltaxx_clsize3", "Xstrip deltaxx clsize=3",  80,   -0.2,   0.2);
    HistInit("XClusterCog", "XCluster Cog",   nx,   0.,xn );
    HistInit("XstripSimHit", "Xstrip SimHit", nx,   0.,xn );
    HistInit("XSimHit", "Xcoord of SimHit",   100,   0.,20. );
    HistInit("XDeltaStrip", "XDelta Strip",   100,   -100.,100. );
    HistInit("XClustSize", "X Cluster Size",   20,   0.,10. );
    HistInit("XAmplitudes", "Xstrip Amplitudes",       100,   0.,100. );
    HistInit("XAmplitudesMax", "Xstrip Amplitudes",    100,   0.,100. );
    HistInit("XAmplitudesRest", "Xstrip Amplitudes",   100,   0.,100. );
    HistInit("XClustFirstStrip", "XCluster First Strip",   nx,   0.,xn );
    HistInit("XClusterErr", "XCluster sigma for width",   100,   0.,0.10 );
    HistInit("XClusterErrLog", "XCluster sigma for width",   100,  -9, 0. );
    // 2D:
    HistInit("X2DSimHitcog","X SimHit cog",     40, 0.0,20.0, 40, 0.0,20.0);
    HistInit("X2DZSimHit","ZX 2D SimHit",      100, 0.,8000., 40, 0.,20.);
//     HistInit("X2DSimHitcogCopy","X SimHit cog",100, 9.0,11.0,100, 9.0,11.0);
    HistInit("X2DSimHitcogCopy","X SimHit cog",100, 0.0,20.0,100, 0.0,20.0);

    // Y:
    HistInit("clnum1Y", "Number Of Clusters in Y 1Station",     11,   0.,   11.);
    HistInit("clnum2Y", "Number Of Clusters in Y 2Station",     11,   0.,   11.);
    HistInit("clnum3Y", "Number Of Clusters in Y 3Station",     11,   0.,   11.);
    HistInit("clnum4Y", "Number Of Clusters in Y 4Station",     11,   0.,   11.);
    HistInit("clnum0Y", "Number Of Clusters in Y all together", 41,   0.,   41.);
    HistInit("Ystrip_deltayy", "Ystrip Dy",                       80, -0.2, 0.2);
    HistInit("Ystrip_deltayy_clsize1", "Ystrip deltayy clsize=1", 80, -0.2, 0.2);
    HistInit("Ystrip_deltayy_clsize2", "Ystrip deltayy clsize=2", 80, -0.2, 0.2);
    HistInit("Ystrip_deltayy_clsize3", "Ystrip deltayy clsize=3", 80, -0.2, 0.2);
    HistInit("Ystrip_deltayy1", "Ystrip1 Dy",    80,   -0.2,   0.2);
    HistInit("Ystrip_deltayy2", "Ystrip2 Dy",    80,   -0.2,   0.2);
    HistInit("Ystrip_deltayy3", "Ystrip3 Dy",    80,   -0.2,   0.2);
    HistInit("Ystrip_deltayy4", "Ystrip4 Dy",    80,   -0.2,   0.2);
    HistInit("YClusterCog", "YCluster Cog",   ny,   0.,yn );
    HistInit("YstripSimHit", "Ystrip SimHit", ny,   0.,yn );
    HistInit("YSimHit", "Ycoord of SimHit",   100,   0.,10. );
    HistInit("YDeltaStrip", "YDelta Strip",   100,   -100.,100. );
    HistInit("YClustSize", "Y Cluster Size",   20,   0.,10. );
    HistInit("YAmplitudes", "Ystrip Amplitudes",       100,   0.,100. );
    HistInit("YAmplitudesMax", "Ystrip Amplitudes",    100,   0.,100. );
    HistInit("YAmplitudesRest", "Ystrip Amplitudes",   100,   0.,100. );
    HistInit("YClustFirstStrip", "YCluster First Strip",   ny,   0.,yn );
    HistInit("YClusterErr", "YCluster sigma for width",   100,   0.,0.10 );
    HistInit("YClusterErrLog", "YCluster sigma for width",   100,  -9, 0. );
    // 2D:
    HistInit("Y2DSimHitcog","Y SimHit cog",     20, 0.0,10.0, 20, 0.0,10.0);
    HistInit("Y2DZSimHit","ZY 2D SimHit",      100, 0.,8000., 20, 0.,10.);
    HistInit("Y2DSimHitcogCopy","Y SimHit cog",100, 4.0,6.0,100, 4.0,6.0);

    // for cl. selection efficiencies:
    int iu01=81; float au01=iu01;
    HistInit("iu0",    "continue numbering",   iu01,   0.,au01 );
    HistInit("iu1",    "iu if cluster exist",   iu01,   0.,au01 );

    // track reconstruction:
    //X
    // HistInit("dXinVtxTrack","dX in Vtx for RecTrack",100, -0.1,0.1);
    // HistInit("nhitplanesX","number of hit planes in X",41, 0.,41.);
    // HistInit("chisqX","chisq X",25, 0.,5.);
    //X1
    HistInit("d1XinVtxTrack","d1X in Vtx for RecTrack",100, -0.5,0.5);
    HistInit("nhitplanes1X","number of hit planes in 1X",11, 0.,11.);
    HistInit("chisq1X","chisq 1X",30, 0.,3.);
    //X2
    HistInit("d2XinVtxTrack","d2X in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanes2X","number of hit planes in 2X",31, 0.,31.);
    HistInit("chisq2X","chisq 2X",30, 0.,3.);
    //X3
    HistInit("d3XinVtxTrack","d3X in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanes3X","number of hit planes in 3X",31, 0.,31.);
    HistInit("chisq3X","chisq 3X",30, 0.,3.);

    //Y
    //  HistInit("dYinVtxTrack","dY in Vtx for RecTrack",100, -0.1,0.1);
    //  HistInit("nhitplanesY","number of hit planes in Y",41, 0.,41.);
    //  HistInit("chisqY","chisq Y",25, 0.,5.);
    //Y1
    HistInit("d1YinVtxTrack","d1Y in Vtx for RecTrack",100, -0.5,0.5);
    HistInit("nhitplanes1Y","number of hit planes in 1Y",11, 0.,11.);
    HistInit("chisq1Y","chisq 1Y",30, 0.,3.);
    //Y2
    HistInit("d2YinVtxTrack","d2Y in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanes2Y","number of hit planes in 2Y",31, 0.,31.);
    HistInit("chisq2Y","chisq 2Y",30, 0.,3.);
    //Y3
    HistInit("d3YinVtxTrack","d3Y in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanes3Y","number of hit planes in 3Y",31, 0.,31.);
    HistInit("chisq3Y","chisq 3Y",30, 0.,3.);


    // dTheta:(mkrad)
   // HistInit("dthtrack", "dthtrack", 100, -20.,20.);
    HistInit("dthtrack1","dthtrack1",100, -1.,1.);
    HistInit("dthtrack2","dthtrack2",100, -20.,20.);
    HistInit("dthtrack3","dthtrack3",100, -20.,20.);


    HistInit("d1thetax2", "d1thetax2", 100, -20.,20.);
    HistInit("d2thetax2", "d2thetax2", 100, -20.,20.);
    HistInit("d3thetax2", "d3thetax2", 100, -20.,20.);
    HistInit("d4thetax2", "d4thetax2", 100, -20.,20.);
    HistInit("d5thetax2", "d5thetax2", 100, -20.,20.);
    HistInit("d6thetax2", "d6thetax2", 100, -20.,20.);
    HistInit("d7thetax2", "d7thetax2", 100, -20.,20.);
    HistInit("d8thetax2", "d8thetax2", 100, -20.,20.);


    // dPhi:(mkrad)
    //    HistInit("dphitrack","dphitrack",  100, -100.,100.);
    HistInit("dphitrack1","dphitrack1",100, -3.,3.);
    HistInit("dphitrack2","dphitrack2",100, -100.,100.);
    HistInit("dphitrack3","dphitrack3",100, -100.,100.);

    // dThetaX, dThetaY (mkrad):
   // HistInit("dthetax", "dthetax", 100, -20.,20.);
   // HistInit("dthetay", "dthetay", 100, -20.,20.);
    HistInit("dthetax2", "dthetax2", 100, -20.,20.);
    HistInit("dthetay2", "dthetay2", 100, -20.,20.);
    HistInit("dthetax3", "dthetax3", 100, -20.,20.);
    HistInit("dthetay3", "dthetay3", 100, -20.,20.);

    HistInit("R2DTXTXres","2D TX vs TXres",      8, 0.0,0.4, 100, -4.,4.);
    HistInit("R2DCHI2TXres","2D CHI2 vs TXres", 20, 0.6,3.0, 100, -10.,10.);

    // 2D:
    HistInit("2DZYseca","ZY 2D RecHits allsec",100, -500.,8500.,100, 0.,10.);
    HistInit("2DZXseca","ZX 2D RecHits allsec",100, -500.,8500.,100, 0.,20.);

    HistInit("2DZYsec1","ZY 2D RecHits secti1",100, -200.,500.,100, 0.,10.);
    HistInit("2DZXsec1","ZX 2D RecHits secti1",100, -200.,500.,100, 0.,20.);

// plots for tracl collection START

    //XY tracks selected
    HistInit("numberOfXandYtracks","numberOfXandYtracks",10, 0.,5.);
    HistInit("ntrackscoll","ntrackscoll",10, 0.,  5.);
    // dPhi:(mkrad)
    HistInit("dphitrack","dphitrack",  100, -100.,100.);
    // dThetaX, dThetaY (mkrad):
    HistInit("dthetax", "dthetax", 100, -20.,20.);
    HistInit("dthetay", "dthetay", 100, -20.,20.);
    // dTheta:(mkrad)
    HistInit("dthtrack", "dthtrack", 100, -20.,20.);
    //X
    HistInit("dXinVtxTrack","dX in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanesX","number of hit planes in X",31, 0.,31.);
    HistInit("chisqX","chisq X",30, 0.,3.);
    //Y
    HistInit("dYinVtxTrack","dY in Vtx for RecTrack",100, -0.1,0.1);
    HistInit("nhitplanesY","number of hit planes in Y",25, 0.,25.);
    HistInit("chisqY","chisq Y",30, 0.,3.);

// test with track collection
    HistInit("tocollection", "tocollection",  60, -0.2,0.2);
//    HistInit("tocollection0","tocollection0", 60, -6.0,6.0);
    HistInit("tocollection0","tocollection0", 60, -0.6,0.6);
    HistInit("tocollection1","tocollection1", 60, -0.2,0.2);
    HistInit("tocollection2","tocollection2", 60, -0.2,0.2);

    HistInit("ntocollection", "ntocollection",  10, -0.15,0.15);
    //HistInit("ntocollection0","ntocollection0", 20, -5.5,5.5);
    HistInit("ntocollection0","ntocollection0", 13, -0.65,0.65);
    HistInit("ntocollection1","ntocollection1", 7, -0.14,0.14);
    HistInit("ntocollection2","ntocollection2", 7, -0.14,0.14);

    HistInit("stocollection", "stocollection",   5, 0.,0.10);
    // HistInit("stocollection0","stocollection0", 10, 0.,4.0);
    HistInit("stocollection0","stocollection0", 10, 0.,0.60);
    HistInit("stocollection1","stocollection1", 10, 0.,0.20);
    HistInit("stocollection2","stocollection2", 10, 0.,0.20);
// plots for track collection END

    HistInit("efftracktheta4", "efftracktheta4",    10,  0.,0.4);
    HistInit("efftracktheta",  "efftracktheta",     10,  0.,0.4);
    HistInit("eff2tracktheta4","eff2tracktheta4",   10,  0.,0.4);
    HistInit("eff2tracktheta", "eff2tracktheta",    10,  0.,0.4);
    HistInit("averdthetavsd12", "averdthetavsd12",  20,  0.,20.0);
    HistInit("eff1trackdref124", "eff1trackdref124",20,  0.,20.0);
    HistInit("efftrackdref124", "efftrackdref124",  20,  0.,20.0);
    HistInit("efftrackdref12", "efftrackdref12",    20,  0.,20.0);

    HistInit("effnhitplanesX4", "effnhitplanesX4", 25, 0.,25.);
    HistInit("effnhitplanesX" , "effnhitplanesX",  25, 0.,25.);
    HistInit("effnhitplanes2X4","effnhitplanes2X4",25, 0.,25.);
    HistInit("effnhitplanes2X" ,"effnhitplanes2X", 25, 0.,25.);
    HistInit("effnhitplanes3X4","effnhitplanes3X4",25, 0.,25.);
    HistInit("effnhitplanes3X" ,"effnhitplanes3X", 25, 0.,25.);

    HistInit("clnumX2Ttacks1Sec4" ,"clnumX2Ttacks1Sec4",  8, 2.,10.);
    HistInit("clnumX2Ttacks2Sec4" ,"clnumX2Ttacks2Sec4",  8, 2.,10.);
    HistInit("clnumX2Ttacks3Sec4" ,"clnumX2Ttacks3Sec4",  8, 2.,10.);
    HistInit("clnumX2Ttacks1Sec"  ,"clnumX2Ttacks1Sec",   8, 2.,10.);
    HistInit("clnumX2Ttacks2Sec"  ,"clnumX2Ttacks2Sec",   8, 2.,10.);
    HistInit("clnumX2Ttacks3Sec"  ,"clnumX2Ttacks3Sec",   8, 2.,10.);

    HistInit("losthitsX2Dnhit" ,"losthitsX2Dnhit", 20, 5.,25.);
    HistInit("losthitsX2D" ,    "losthitsX2D",     20, 5.,25.);

    HistInit("losthitsX3Dnhit" ,"losthitsX3Dnhit", 31, 0.,31.);
    HistInit("losthitsX3D" ,    "losthitsX3D",     31, 0.,31.);

    HistInit("dref12", "dref12",  40,  0.,20.0);
    HistInit("drefy12", "drefy12",  20,  0.,10.0);
    HistInit("xref", "xref",  50,  0.,-50.0);
    HistInit("xref2", "xref2",  50,  0.,-50.0);
    HistInit("mintheta", "mintheta",  100, -20.,20.);
    HistInit("mintheta2", "mintheta2",  100, -20.,20.);
    HistInit("minthetay", "minthetay",  100, -20.,20.);
    HistInit("minthetay2", "minthetay2",  100, -20.,20.);
    HistInit("ccchindfx", "ccchindfx",  30, 0.,3.);
    HistInit("ccchindfx2", "ccchindfx2",  30, 0.,3.);
    HistInit("yref", "yref",  50,  0.,-50.0);
    HistInit("yref2", "yref2",  50,  0.,-50.0);
    HistInit("thetaXmrad", "thetaXmrad",  50,  0.,0.50);
    HistInit("thetaX2mrad", "thetaX2mrad",  40,  0.,0.40);

    HistInit("xxxxxx", "xxxxxx",      90,  -1.,  1.0);
    HistInit("yyyyyy", "yyyyyy",      90,  -1.,  1.0);
    HistInit("xxxxxxeq", "xxxxxxeq",  90,  -1.,  1.0);
    HistInit("yyyyyyeq", "yyyyyyeq",  90,  -1.,  1.0);
    HistInit("xxxxxxno", "xxxxxxno",  90,  -1.,  1.0);
    HistInit("yyyyyyno", "yyyyyyno",  90,  -1.,  1.0);

    HistInit("xxxxxxs", "xxxxxxs",      80,  -22.2,-21.8);
    HistInit("yyyyyys", "yyyyyys",      80,  -0.2,0.2);
    HistInit("xxxxxxeqs", "xxxxxxeqs",  80,  -22.2,-21.8);
    HistInit("yyyyyyeqs", "yyyyyyeqs",  80,  -0.2,0.2);
    HistInit("xxxxxxnos", "xxxxxxnos",  80,  -22.2,-21.8);
    HistInit("yyyyyynos", "yyyyyynos",  80,  -0.2,0.2);
    HistInit("dthdiff", "dthdiff",       100,  -0.25,0.25);
    HistInit("dthdiffeq", "dthdiffeq",   100,  -0.25,0.25);
    HistInit("dthdiffno", "dthdiffno",   100,  -0.25,0.25);
    HistInit("dthdif", "dthdif",        75,  0.,0.25);
    HistInit("dthdifeq", "dthdifeq",    75,  0.,0.25);
    HistInit("dthdifno", "dthdifno",    75,  0.,0.25);


    HistInit("mindthtrack", "mindthtrack",  100, -20.,20.);
    HistInit("mindphitrack","mindphitrack",  100, -100.,100.);
    HistInit("minthtrue", "minthtrue",  50,  0.,0.50);
    HistInit("minthreal", "minthreal",  50,  0.,0.50);
    HistInit("mindthtrack2", "mindthtrack2",  100, -20.,20.);
    HistInit("mindphitrack2","mindphitrack2",  100, -100.,100.);
    HistInit("minthtrue2", "minthtrue2",  50,  0.,0.50);
    HistInit("minthreal2", "minthreal2",  50,  0.,0.50);
//
//
    HistInit("ATest", "ATest",  100,  0.,100);

//
//
    HistInit("ZZZall", "ZZZall",  100,  0.,420000);
    HistInit("ZZZ420", "ZZZ420",  100,  400000.,430000);
    HistInit("XXX420", "XXX420",  100,  -20.,20);
    HistInit("YYY420", "YYY420",  100,  -10.,10);
    HistInit("npart420", "npart420",  10,  0.,10.);

    HistInit("2DXY420", "2DXY420",    100, -25.,5.,100, -5.,5.);
    HistInit("2DXY420Tr", "2DXY420Tr",100, -25.,5.,100, -5.,5.);
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

        fHistArray->Write();
        file->Close();
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
  //  G4int         parentID       = theTrack->GetParentID();   //   !!!
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
	SumEnerDeposit += EnerDeposit;
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
      

      if(trackmom.z() > 100000. && curstepnumber == 1) {
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
  
  
#ifdef ddebugprim
      std::cout << " -------------------------------------------------------------" << std::endl;
      std::cout << " -------------------------------------------------------------" << std::endl;
      std::cout << " -------------------------------------------------------------" << std::endl;
#endif
  // prim.vertex:
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  
#ifdef ddebugprim
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
#ifdef ddebugprim
  std::cout << "zmilimit= :" << zmilimit << std::endl;
#endif
// ==========================================================================================loop over vertexies
  double zref=-100., xref=-100., yref=-100., bxtrue=-100., bytrue=-100.,dref12=-100.,drefy12=-100.;
    //ref = z1+8000.;     // info: center of 1st station at 0.
  double       xref2=-100., yref2=-100., bxtrue2=-100., bytrue2=-100.;
  double       xref3=-100., yref3=-100., bxtrue3=-100., bytrue3=-100.;
  double ZZZ420=-999999;
  for (int iv = 0 ; iv<nvertex; ++iv) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(iv);
    if (avertex == 0) std::cout<<"DigFP420Test:End Of Event ERR: pointer to vertex = 0"<< std::endl;
    G4int npart = avertex->GetNumberOfParticle();
    TheHistManager->GetHisto("ZZZall")->Fill(avertex->GetZ0());

    if(avertex->GetZ0() < 400000.){
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
#ifdef ddebugprim
      std::cout << "Vertex number :" <<iv << std::endl;
      std::cout << "Vertex Z= :" <<(*evt)()->GetPrimaryVertex(iv)->GetZ0() << std::endl;
      std::cout << "Vertex X= :" <<(*evt)()->GetPrimaryVertex(iv)->GetX0() << std::endl;
      std::cout << "Vertex Y= :" <<(*evt)()->GetPrimaryVertex(iv)->GetY0() << std::endl;
      
#endif
    TheHistManager->GetHisto("ZZZ420")->Fill(ZZZ420);
    XXX420 = avertex->GetX0();
    YYY420 = avertex->GetY0();
    ZZZ420 = avertex->GetZ0();
    zref =    8000. + z420 - ZZZ420;     // info: center of 1st station at 0.
    TheHistManager->GetHisto("XXX420")->Fill(XXX420);
    TheHistManager->GetHisto("YYY420")->Fill(YYY420);
    TheHistManager->GetHisto2("2DXY420")->Fill(XXX420,YYY420);
    TheHistManager->GetHisto("npart420")->Fill(float(npart));
    if(npart !=1)std::cout << "DigFP420Test::warning: NumberOfPrimaryPart != 1--> = " <<npart<<std::endl;
#ifdef ddebugprim
    std::cout << "number of particles for Vertex = " << npart << std::endl;
#endif
    if (npart==0)std::cout << "DigFP420Test: End Of Event ERR: no NumberOfParticle" << std::endl;
    
    // =============================================================loop over particles of ZZZ420 vertex
    for (int i = 0 ; i<npart; ++i) {
#ifdef ddebugprim
      std::cout << " -------------------------" << std::endl;
#endif
      thePrim=avertex->GetPrimary(i);
      G4ThreeVector   mom  = thePrim->GetMomentum();
      // =====================================
      //  avertex->GetTotalEnergy()    mom.mag()   mom.t()    mom.vect().mag() 
//    std::cout << "mom.mag() = " << mom.mag() << std::endl;
////    std::cout << "mom.t() = " << mom.t() << std::endl;
////    std::cout << "mom.vect().mag() = " << mom.vect().mag() << std::endl;
////    std::cout << "thePrim->GetTotalEnergy() = " << thePrim->GetTotalEnergy() << std::endl;
      // =====================================
      if(i==0){
	phi = mom.phi();
	if (phi < 0.) phi += twopi;
	phigrad = phi*180./pi;
	th     = mom.theta();
	eta = -log(tan(th/2));
	xi = 1.-mom.mag()/7000000.;

	bxtrue = tan(th)*cos(phi);
	bytrue = tan(th)*sin(phi);

	xref = vx + (zref-vz)*bxtrue;
	yref = vy + (zref-vz)*bytrue;
#ifdef ddebugprim
	std::cout << "DigFP420Test: vx = " << XXX420 << " th=" << th << " phi=" << phi << " xref=" << xref << std::endl;
	std::cout << " tan(th) = " << tan(th) << " cos(phi)=" << cos(phi) << " bxtrue=" << bxtrue << std::endl;
#endif
	//  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
#ifdef ddebugprim
	std::cout << " lastpo.x()=" << lastpo.x() << std::endl;
	std::cout << " lastpo.y()=" << lastpo.y() << std::endl;
	std::cout << " lastpo.z()=" << lastpo.z() << std::endl;
#endif
	if(  lastpo.z()< zmilimit ) {
	  varia = 1;
	}
	else{
	  varia = 2;
	} 
	
      }
      else if(i==1){
	phi2= mom.phi();
	if (phi2< 0.) phi2 += twopi;
	phigrad2 = phi2*180./pi;
	th2     = mom.theta();
	eta2 = -log(tan(th2/2));
        xi2 = 1.-mom.mag()/7000000.;
	// 2st primary track 
	bxtrue2= tan(th2)*cos(phi2);
	bytrue2= tan(th2)*sin(phi2);
	xref2= vx + (zref-vz)*bxtrue2;
	yref2= vy + (zref-vz)*bytrue2;
#ifdef ddebugprim
	std::cout << "DigFP420Test: vx = " <<  XXX420<< " th2=" << th2 << " phi2=" << phi2 << " xref2=" << xref2 << std::endl;
	std::cout << " tan(th2) = " << tan(th2) << " cos(phi2)=" << cos(phi2) << " bxtrue2=" << bxtrue2 << std::endl;
#endif
	
	//  if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	if(  lastpo.z()< zmilimit  ) {
	  varia2= 1;
	}
	else{
	  varia2= 2;
	} 
	
      }
      else if(i==2){
	phi3 = mom.phi();
	if (phi3 < 0.) phi3 += twopi;
	phigrad3 = phi3*180./pi;
	th3     = mom.theta();
	eta3 = -log(tan(th3/2));
        xi3 = 1.-mom.mag()/7000000.;
	// 3rd primary track 
	bxtrue3= tan(th3)*cos(phi3);
	bytrue3= tan(th3)*sin(phi3);
	xref3= vx + (zref-vz)*bxtrue3;
	yref3= vy + (zref-vz)*bytrue3;
	

	if(  lastpo.z()< zmilimit || (lastpo.z()>zmilimit && lastpo.perp()> 100.) ) {
	  varia3= 1;
	}
	else{
	  varia3= 2;
	} 
	
      }
      else {
	std::cout << "DigFP420Test:WARNING i>3" << std::endl; 
      }// if(i
      // =====================================
      
#ifdef ddebugprim0
      std::cout << " i=" << i << "DigFP420Test: at 420m mom = " << mom 
		<< std::endl;
#endif
      // primary vertex:
      //	 G4double vx=0.,vy=0.,vz=0.;
      vx = avertex->GetX0();
      vy = avertex->GetY0();
      vz = avertex->GetZ0();
      TheHistManager->GetHisto("VtxX")->Fill(vx);
      TheHistManager->GetHisto("VtxY")->Fill(vy);
      TheHistManager->GetHisto("VtxZ")->Fill(vz);
#ifdef ddebugprim
      std::cout << " -------------------------------------------------------------" << std::endl;
      std::cout << "DigFP420Test: Vertex vx = " << vx << " vy=" << vy << " vz=" << vz << std::endl;
      std::cout << " Vertex vx = " << vx << " vy=" << vy << "vz=" << vz << std::endl;
      std::cout << " varia = " << varia << " varia2=" << varia2 << " i=" << i << std::endl;
#endif
    }// loop over particles of ZZZ420 vertex  (int i



    //                                                                              .
    dref12 = abs(xref2 - xref);
    drefy12 = abs(yref2 - yref);
#ifdef ddebugprim
    std::cout << " dref12 = " << dref12 << std::endl;
#endif
    

    }//if(fabs(ZZZ420)
  }// prim.vertex loop end
  //                                                                              preparations:
//temporary:
//  eta = eta0;
//  xi = xi0;
    TheHistManager->GetHisto("PrimaryXi")->Fill(xi);
    TheHistManager->GetHisto("PrimaryXiLog")->Fill(TMath::Log10(xi));
    TheHistManager->GetHisto("PrimaryEta")->Fill(eta);
    //                                                                              .
    TheHistManager->GetHisto("xref")->Fill(xref);
    TheHistManager->GetHisto("xref2")->Fill(xref2);
    TheHistManager->GetHisto("dref12")->Fill(dref12);
    TheHistManager->GetHisto("drefy12")->Fill(drefy12);
    TheHistManager->GetHisto("yref")->Fill(yref);
    TheHistManager->GetHisto("yref2")->Fill(yref2);
    TheHistManager->GetHisto("thetaXmrad")->Fill(fabs(bxtrue)*1000.);
    //	TheHistManager->GetHisto("thetaX2mrad")->Fill(fabs(bxtrue2)*1000.);
    TheHistManager->GetHisto("thetaX2mrad")->Fill(fabs(bxtrue)*1000.);

    TheHistManager->GetHisto("PrimaryPhigrad")->Fill(phigrad);
    // TheHistManager->GetHisto("PrimaryTh")->Fill(th*180./pi); / dergee
    TheHistManager->GetHisto("PrimaryTh")->Fill(th*1000.);// mlrad
    
    TheHistManager->GetHisto("PrimaryLastpoZ")->Fill(lastpo.z());
    if(lastpo.z() <  z4  ) {
      TheHistManager->GetHisto("PrimaryLastpoX")->Fill(lastpo.x());
      TheHistManager->GetHisto("PrimaryLastpoY")->Fill(lastpo.y());
    }
    if(numofpart >  4  ) {
      TheHistManager->GetHisto("XLastpoNumofpart")->Fill(lastpo.x());
      TheHistManager->GetHisto("YLastpoNumofpart")->Fill(lastpo.y());
    }


//=========================== thePrim != 0 ================================================================================
//    if (thePrim != 0   && vz < -20.) {

		  
//ask 1 tracks	  	  
		  
	if ( thePrim != 0 && ZZZ420 != -999999.
			   ) {
	  

//	     &&	varia == 2  
//	     && ((xref > -32. && xref < -12.) && (yref > -5. && yref < 5.))  
//	     &&	varia == 2  
//	     && ( fabs(bxtrue)*1000. > 0.1  && fabs(bxtrue)*1000.<0.4 )
		  
// ask 2 tracks		  
		  
	  	  
		  
	/*	  
	if ( thePrim != 0  && ZZZ420 != -999999.
	     && ((xref  > -32. && xref  < -12.) && (yref  > -5. && yref  < 5.))  
	     && ((xref2 > -32. && xref2 < -12.) && (yref2 > -5. && yref2 < 5.))  
	     && dref12 > 1.0 && drefy12 > 1.0       
	     ) {
	  */
	  
	  
	   //  &&	( varia == 2 && varia2 == 2 ) 
	   //  && dref12 > 1.       
	  //	     && (( fabs(bxtrue)*1000.>0.1)&&( fabs(bxtrue)*1000.<0.4) ) || (( fabs(bxtrue2)*1000. > 0.1)&&( fabs(bxtrue2)*1000.<0.4) )  
	  
	  
	  /////////////////////////////////////////////////////////////////
	  //      unsigned int clnumcut=1;// ask 2 tracks
	  //          unsigned int clnumcut=0;//ask 1 tracks
	  /////////////////////////////////////////////////////////////////
	    


  // ==========================================================================

  // hit map for FP420
  // ==================================

  map<int,float,less<int> > themap;
  map<int,float,less<int> > themap1;

  map<int,float,less<int> > themapxystrip;
  map<int,float,less<int> > themapxy;
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





  varia = 0;
  if( varia == 0  ) {
    
    double  totallosenergy= 0.;
    int AATest[80];
    for (int j=0; j<80; j++) {
      AATest[j]=0;
    }

   for (int j=0; j<theCAFI->entries(); j++) {
    FP420G4Hit* aHit = (*theCAFI)[j];

    Hep3Vector hitEntryLocalPoint = aHit->getEntryLocalP();
    Hep3Vector hitExitLocalPoint = aHit->getExitLocalP();
    Hep3Vector hitPoint = aHit->getEntry();
    unsigned int unitID = aHit->getUnitID();
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
*/


    double  losenergy = aHit->getEnergyLoss();


    /*
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

    // zside=1,2 ; zmodule=1,10 ; sector=1,3
    if(zside==0||sector==0||zmodule==0){
    std::cout << "DigFP420Test:ERROR: zside = " << zside  << " sector = " << sector  << " zmodule = " << zmodule  << " det = " << det  << std::endl;
    }

    double kplane = -(pn0-1)/2+(zmodule-1); 
    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = -150. + z420 +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  

    if(zside==1){
      zcurrent += (ZGapLDet+ZSiDetL/2);
    }
    if(zside==2){
      zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
                }     

//=======================================
// SimHit position in Local reference frame - middle :
      G4ThreeVector middle = (hitExitLocalPoint+hitEntryLocalPoint)/2.;
      G4ThreeVector mid = (hitExitLocalPoint-hitEntryLocalPoint)/2.;

//
      if (verbosity > 2) {
	std::cout << "DigFP420Test:check " << std::endl;
	std::cout << " zside = " <<zside<< " sector = " <<sector<< " zmodule = " << zmodule<< std::endl;
	std::cout << " hitPoint.z()+ mid.z() = " <<  double (hitPoint.z()+ mid.z()-z420) << std::endl;
	std::cout << " zcurrent = " << double (zcurrent-z420) << " det = " << det << std::endl;
	std::cout << " diff = " << double (hitPoint.z()+ mid.z()- zcurrent) << std::endl;
      }
//=======================================
     //   themapz[unitID]  = hitPoint.z()+ mid.z(); // this line just for studies
      themapz[unitID]  = zcurrent;// finally must be this line !!!

//=======================================

      themapxystrip[unitID] = -1.;// charge in strip coord 

      float numStrips,pitch;
//=======================================
// Y global
	   if(zside==1) {
	     //UserNtuples->fillg24(losenergy,1.);
	     if(losenergy > 0.00003) {
	       themap1[unitID] += 1.;
	     }
// E field (and p+,n+sides) along X,  but define Y coordinate -->  number of strips of the same side(say p+):200*pitchX=20mm
	     numStrips = numStripsY;
	     pitch=pitchY;
	     
             themapxystrip[unitID] = 0.5*(numStrips-1) + middle.x()/pitch ;// charge in strip coord 
	     themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.x();// hit coordinate in l.r.f starting at bot edge of plate
	   }
	   //X
	   if(zside==2){
	     if(losenergy > 0.00003) {
	       themap1[unitID] += 1.;
	     }
	     numStrips = numStripsX;
	     pitch=pitchX;
             themapxystrip[unitID] = 0.5*(numStrips-1) + middle.y()/pitch ;// charge in strip coord 

	     themapxy[unitID]  = (numStrips-1)*pitch/2. + middle.y(); // hit coordinate in l.r.f starting at left edge of plate 


    } // MIonly or noMIonly ENDED
//    }

   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!

   }  // for loop on all hits ENDED  ENDED  ENDED  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!
   //     !!!!!!!!!!!!!

   for (int j=0; j<80 && AATest[j]!=0. ; j++) {
     TheHistManager->GetHisto("ATest")->Fill(AATest[j]);
   }



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
                                                                                                 
          
// ==============================================================================================   produce Digi start
//    produce();

   theDigitizerFP420->produce(theCAFI,output);

    if (verbosity > 2) {
      std::cout <<" ===== RecFP420Test:: access to DigiCollectionFP420 " << std::endl;
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
    if(zside==1){
      //int iy= istrip.channel()/numStripsY;
      //int ix= istrip.channel() - iy*numStripsY;
      int iy= istrip.stripHW();
      int ix= istrip.stripH();
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
      if(sector==1){
	TheHistManager->GetHisto("DigiYstripAdcSigma1")->Fill(istrip.adc()/sigmanoise);
      }
      else if(sector==2){
	TheHistManager->GetHisto("DigiYstripAdcSigma2")->Fill(istrip.adc()/sigmanoise);
      }
      else if(sector==3){
	TheHistManager->GetHisto("DigiYstripAdcSigma3")->Fill(istrip.adc()/sigmanoise);
      }
      else if(sector==4){
	TheHistManager->GetHisto("DigiYstripAdcSigma4")->Fill(istrip.adc()/sigmanoise);
      }
    }
    // X:
    else if(zside==2){
          //int iy= istrip.channel()/numStripsX;
	  //  int ix= istrip.channel() - iy*numStripsX;
      int iy= istrip.stripVW();
      int ix= istrip.stripV();
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
      else if(sector==4){
	TheHistManager->GetHisto("DigiXstripAdcSigma4")->Fill(istrip.adc()/sigmanoise);
      }
    }
  //==================================
#ifdef mydigidebug10
       if(zside == 2) {
  std::cout << " RecFP420Test::check: HDigiFP420::  " << std::endl;
  // std::cout << " strip number = " << (*simHitIter).strip() << "  adc = " << (*simHitIter).adc() << std::endl; // std::cout << " strip number = " << simHitIter->strip() << "  adc = " << simHitIter->adc() << std::endl;
  std::cout << " strip number = " << istrip.strip() << "  adc = " << istrip.adc() << std::endl;
  std::cout << " channel = " << istrip.channel() << std::endl;
  int numStrips = numStripsY;
  if(zside==2) numStrips = numStripsX;
  
  //int iy= istrip.channel()/numStrips;
  //int ix= istrip.channel() - iy*numStrips;
  int iy= istrip.stripVW();
  int ix= istrip.stripV();
    // istrip.channel() = iy*numStrips + (ix+k) ;  // Get index 
    std::cout << " ix = " << ix << " iy = " << iy << std::endl;
  
  // std::cout << " strip number = " << astrip->strip() << "  adc = " << astrip->adc() << std::endl;
      }
#endif
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
  
  //     end of check of access to the strip collection
// =======================================================================================check of access to strip collection






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

