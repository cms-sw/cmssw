// Original Author:  Loic Quertenmont

#ifndef HSCP_ANALYSIS_GLOBAL
#define HSCP_ANALYSIS_GLOBAL

//Include widely used in all the codes
#include <string>
#include <vector>
#include <fstream>
#include "TROOT.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TCutG.h" 
#include "TDCacheFile.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TMath.h"
#include "TMultiGraph.h"
#include "TObject.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TRandom3.h"
#include "TTree.h"

//This code is there to enable/disable year dependent code
//#define ANALYSIS2011

double IntegratedLuminosity7TeV = 5003;
double IntegratedLuminosity8TeV = 16319;//11564;

#ifdef ANALYSIS2011
double               SQRTS          = 7;
int                  RunningPeriods = 2;
double               IntegratedLuminosity = IntegratedLuminosity7TeV; 
double               IntegratedLuminosityBeforeTriggerChange = 409.91; 
#else
double               SQRTS          = 8;
int                  RunningPeriods = 1;
double               IntegratedLuminosity = IntegratedLuminosity8TeV;
double               IntegratedLuminosityBeforeTriggerChange = 0;
double               IntegratedLuminosityHigherMETThreshold = 698.991;
#endif

double IntegratedLuminosityFromE(double SQRTS_){
   if(SQRTS_==7){return IntegratedLuminosity7TeV;}
   else if(SQRTS_==8){return IntegratedLuminosity8TeV;}
   else if(SQRTS_==78 || SQRTS_==87){return IntegratedLuminosity7TeV+IntegratedLuminosity8TeV;}
   else{return -1;}
}


// Type of the analysis
int		   TypeMode         = 0; //0 = Tracker-Only analysis (used in 2010 and 2011 papers)
					 //1 = Tracker+Muon analysis (used in 2010 paper)
                                         //2 = Tracker+TOF  analysis (used in 2011 paper)
                                         //3 = TOF Only     analysis (to be used in 2012 paper)
                                         //4 = Q>1          analysis (to be used in 2012 paper)
                                         //5 = Q<1          analysis (to be used in 2012 paper)
                                         //? do not hesitate to define your own --> TOF-Only, mCHAMPs, fractional charge

// directory where to find the EDM files --> check the function at the end of this file, to see how it is defined interactively
std::string BaseDirectory = "undefined... Did you call InitBaseDirectory() ? --> ";


// binning for the pT, mass, and IP distributions
double             PtHistoUpperBound   = 1200;
double             MassHistoUpperBound = 2000;
int		   MassNBins           = 200;
double             IPbound             = 1.0;

// Thresholds for candidate preselection --> note that some of the followings can be replaced by Analysis_Step3 function arguments
double             GlobalMaxEta     =   1.5;    // cut on inner tracker track eta
double             GlobalMaxV3D     =   99999;//0.50;   // cut on 3D distance (cm) to closest vertex
double             GlobalMaxDZ      =   0.50;   // cut on 1D distance (cm) to closest vertex in "Z" direction
double             GlobalMaxDXY     =   0.50;   // cut on 2D distance (cm) to closest vertex in "R" direction
double             GlobalMaxChi2    =   5.0;    // cut on Track maximal Chi2/NDF
int                GlobalMinQual    =   2;      // cut on track quality (2 meaning HighPurity tracks)
unsigned int       GlobalMinNOH     =   8;      // cut on number of (valid) track pixel+strip hits 
int                GlobalMinNOPH    =   2;      // cut on number of (valid) track pixel hits 
double             GlobalMinFOVH    =   0.8;    // cut on fraction of valid track hits
unsigned int       GlobalMinNOM     =   6;      // cut on number of dEdx hits (generally equal to #strip+#pixel-#ClusterCleaned hits, but this depend on estimator used)
double             GlobalMinNDOF    =   8;      // cut on number of     DegreeOfFreedom used for muon TOF measurement
double             GlobalMinNDOFDT  =   6;      // cut on number of DT  DegreeOfFreedom used for muon TOF measurement
double             GlobalMinNDOFCSC =   6;      // cut on number of CSC DegreeOfFreedom used for muon TOF measurement
double             GlobalMaxTOFErr  =   0.07;   // cut on error on muon TOF measurement
double             GlobalMaxPterr   =   0.25;   // cut on error on track pT measurement 
double             GlobalMaxTIsol   =  50;      // cut on tracker isolation (SumPt)
double             GlobalMaxRelTIsol   =  9999999; // cut on relative tracker isolation (SumPt/Pt)
double             GlobalMaxEIsol   =  0.30;   // cut on calorimeter isolation (E/P)
double             GlobalMinPt      =  45.00;   // cut on pT    at PRE-SELECTION
double             GlobalMinIs      =   0.0;    // cut on dEdxS at PRE-SELECTION (dEdxS is generally a  discriminator)
double             GlobalMinIm      =   3.0;    // cut on dEdxM at PRE-SELECTION (dEdxM is generally an estimator    )
double             GlobalMinTOF     =   1.0;    // cut on TOF   at PRE-SELECTION
const int          MaxPredBins      =   6;      //The maximum number of different bins prediction is done in for any of the analyses (defines array size)
int                PredBins         =   0;      //How many different bins the prediction is split in for analysis being run, sets how many histograms are actually initialized.

// dEdx related variables, Name of dEdx estimator/discriminator to be used for selection (dEdxS) and for mass reconstruction (dEdxM)
// as well as the range for the dEdx variable and K/C constant for mass reconstruction
std::string        dEdxS_Label     = "dedxASmi";
double             dEdxS_UpLim     = 1.0;
std::string        dEdxS_Legend    = "I_{as}";
std::string        dEdxM_Label     = "dedxHarm2";
double             dEdxM_UpLim     = 15.0;
std::string        dEdxM_Legend    = "I_{h} (MeV/cm)";
double             dEdxK_Data      = 2.529;
double             dEdxC_Data      = 2.772;
double             dEdxK_MC        = 2.529;
double             dEdxC_MC        = 2.772;

// TOF object to be used for combined, DT and CSC TOF measurement
std::string        TOF_Label       = "combined";
std::string        TOFdt_Label     = "dt";
std::string        TOFcsc_Label    = "csc";

//Variables used in the TOF only HSCP search
float              DTRegion      =   0.9;  //Define the dividing line between DT and 
float              CSCRegion     =   0.9;  //CSC regions of CMS
float              CosmicMinDz   =   70.;  //Min dz displacement to be tagged as cosmic muon
float              CosmicMaxDz   =   120.; //Max dz displacement for cosmic tagged tracks
//double             MaxDistTrigger=   0.4;  //Max Dist to trigger object
double             minSegEtaSep  = 0.1;   //Minimum eta separation between SA track and muon segment on opposite side of detector
const int          DzRegions     = 6;      //Number of different Dz side regions used to make cosmic background prediction
int                minMuStations = 2;


//for initializing PileupReweighting utility.
const   float TrueDist2011_f[35] = {0.00285942, 0.0125603, 0.0299631, 0.051313, 0.0709713, 0.0847864, 0.0914627, 0.0919255, 0.0879994, 0.0814127, 0.0733995, 0.0647191, 0.0558327, 0.0470663, 0.0386988, 0.0309811, 0.0241175, 0.018241, 0.0133997, 0.00956071, 0.00662814, 0.00446735, 0.00292946, 0.00187057, 0.00116414, 0.000706805, 0.000419059, 0.000242856, 0.0001377, 7.64582e-05, 4.16101e-05, 2.22135e-05, 1.16416e-05, 5.9937e-06, 5.95542e-06};//from 2011 Full dataset

const   float Pileup_MC_Fall11[35]= {1.45346E-01, 6.42802E-02, 6.95255E-02, 6.96747E-02, 6.92955E-02, 6.84997E-02, 6.69528E-02, 6.45515E-02, 6.09865E-02, 5.63323E-02, 5.07322E-02, 4.44681E-02, 3.79205E-02, 3.15131E-02, 2.54220E-02, 2.00184E-02, 1.53776E-02, 1.15387E-02, 8.47608E-03, 6.08715E-03, 4.28255E-03, 2.97185E-03, 2.01918E-03, 1.34490E-03, 8.81587E-04, 5.69954E-04, 3.61493E-04, 2.28692E-04, 1.40791E-04, 8.44606E-05, 5.10204E-05, 3.07802E-05, 1.81401E-05, 1.00201E-05, 5.80004E-06};

const   float Pileup_MC_Summer2012[60] = { 2.560E-06, 5.239E-06, 1.420E-05, 5.005E-05, 1.001E-04, 2.705E-04, 1.999E-03, 6.097E-03, 1.046E-02, 1.383E-02, 1.685E-02, 2.055E-02, 2.572E-02, 3.262E-02, 4.121E-02, 4.977E-02, 5.539E-02, 5.725E-02, 5.607E-02, 5.312E-02, 5.008E-02, 4.763E-02, 4.558E-02, 4.363E-02, 4.159E-02, 3.933E-02, 3.681E-02, 3.406E-02, 3.116E-02, 2.818E-02, 2.519E-02, 2.226E-02, 1.946E-02, 1.682E-02, 1.437E-02, 1.215E-02, 1.016E-02, 8.400E-03, 6.873E-03, 5.564E-03, 4.457E-03, 3.533E-03, 2.772E-03, 2.154E-03, 1.656E-03, 1.261E-03, 9.513E-04, 7.107E-04, 5.259E-04, 3.856E-04, 2.801E-04, 2.017E-04, 1.439E-04, 1.017E-04, 7.126E-05, 4.948E-05, 3.405E-05, 2.322E-05, 1.570E-05, 5.005E-06};

const   float TrueDist2012_f[60] = {1.05858e-06 ,2.79007e-06 ,5.66022e-06 ,2.21761e-05 ,4.333e-05 ,0.00021475 ,0.00127484 ,0.00380513 ,0.00859346 ,0.0164099 ,0.0277558 ,0.0411688 ,0.0518905 ,0.0579633 ,0.0615463 ,0.0640369 ,0.0648159 ,0.0639443 ,0.0622142 ,0.0598481 ,0.0571089 ,0.0543368 ,0.0516781 ,0.0487896 ,0.0449614 ,0.0397967 ,0.0335265 ,0.0267498 ,0.0201118 ,0.0141912 ,0.00941021 ,0.00590948 ,0.00354911 ,0.00204957 ,0.00113529 ,0.000598229 ,0.00029732 ,0.000138844 ,6.11323e-05 ,2.5644e-05 ,1.04009e-05 ,4.139e-06 ,1.63291e-06 ,6.41399e-07 ,2.50663e-07 ,9.71641e-08 ,3.72356e-08 ,1.40768e-08 ,5.24657e-09 ,1.92946e-09 ,7.01358e-10 ,2.52448e-10 ,9.00753e-11 ,3.18556e-11 ,1.11511e-11 ,3.85524e-12 ,1.31312e-12 ,4.3963e-13 ,1.44422e-13 ,4.64971e-14};

// function used to define Axis range and legend automatically from the estimator label
void InitdEdx(std::string dEdxS_Label_){
   if(dEdxS_Label_=="dedxASmi" || dEdxS_Label_=="dedxNPASmi"){
      dEdxS_UpLim  = 1.0;
      dEdxS_Legend = "I_{as High}";
   }else if(dEdxS_Label_=="dedxRASmi" || dEdxS_Label_=="dedxNPRASmi"){
      dEdxS_UpLim  = 1.0;
      dEdxS_Legend = "I_{as Low}";
   }else if(dEdxS_Label_=="dedxProd" || dEdxS_Label_=="dedxNPProd"){
      dEdxS_UpLim  = 1.0;
      dEdxS_Legend = "I_{d}";
   }else{
      dEdxS_UpLim  = 30.0;
      dEdxS_Legend = "I_{h} (MeV/cm)";
   }
}

// function used to define interactively the directory containing the EDM files
// you are please to add the line for your case and not touch the line of the other users
void InitBaseDirectory(){  
   char* analystTmp=getenv("USER");
   char* hostTmp   =getenv("HOSTNAME");
   if(!hostTmp||!analystTmp)return;
   string analyst(analystTmp);
   string host   (hostTmp);
   if(getenv("PWD")!=NULL)host+=string(" PWD=") + getenv("PWD");

   // BaseDirectory is defined as a function of the host you are running on
   if(host.find("ucl.ac.be")!=std::string::npos){
      //BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_11_01/";
      BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_8/12_08_16/";
   }else if(host.find("cern.ch")!=std::string::npos){
      //BaseDirectory = "rfio:/castor/cern.ch/user/r/rybinska/HSCPEDMFiles/";
      BaseDirectory = "root://eoscms//eos/cms/store/cmst3/user/querten/12_08_30_HSCP_EDMFiles/";
   }else if(host.find("fnal.gov")!=std::string::npos){
     BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/venkat12/HSCPEDMFiles/20120901/";
   }else{
      BaseDirectory = "dcache:/pnfs/cms/WAX/11/store/user/venkat12/2012Data/";
      printf("YOUR MACHINE (%s) IS NOT KNOW --> please add your machine to the 'InitBaseDirectory' function of 'Analysis_Global.h'\n", host.c_str());
      printf("HOST=%s  USER=%s\n",host.c_str(), analyst.c_str());
      printf("In the mean time, the directory containing the HSCP EDM file is assumed to be %s\n",BaseDirectory.c_str());
   }

   // BaseDirectory is defined a function of the username
//   if(analyst.find("querten")!=std::string::npos && host.find("ucl.ac.be")!=std::string::npos){
//      BaseDirectory = "/storage/data/cms/users/quertenmont/HSCP/CMSSW_4_2_3/11_11_01/";
//   }   
}




#endif
