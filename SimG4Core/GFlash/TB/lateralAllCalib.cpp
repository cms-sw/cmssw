// Lateral shower development studies - without integration along phi
// To be compiled using the makefile
// 2 loops to get the maximum containment point, 
// 3rd loop to do the analysis (in +- 2mm arount it)

// to run on calibrated rechits reduced trees

/*
  dwjang recipe to run this file

###### data section
#
#foreach imom ($mom)
#    echo "lateralAllCalib input_tb_${imom}.list tb_${imom} $imom 0 0"
#    ./lateralAllCalib input_tb_${imom}.list tb_${imom} $imom 0 0 >&! log.tb_${imom}
#    echo "Done."
#end
#
###### geant4
#
#foreach imom ($mom)
#    echo "lateralAllCalib input_g4_${imom}.list g4_${imom} $imom 0 1"
#    ./lateralAllCalib input_g4_${imom}.list g4_${imom} $imom 0 1 >&! log.g4_${imom}
#    echo "Done."
#end
#
##### gflash

#foreach imom ($mom)
#    echo "lateralAllCalib input_gf_${imom}.list gf_${imom} $imom 0 1"
#    ./lateralAllCalib input_gf_${imom}.list gf_${imom} $imom 0 1 >&! log.gf_${imom}
#    echo "Done."
#end


*/


//! c++ includes              
#include <string>
#include <iostream>
#include <unistd.h>
#include <cmath>
#include <sstream>
#include <fstream>
#include <map>

//! ROOT includes        
#include "TROOT.h"
#include "TFile.h"
#include "TSystem.h"
#include "TChain.h"
#include "TBranch.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"
#include "TMinuit.h"
#include "TString.h"

using namespace std;

//! my includes
#include "fitMaximumPoint.cc"
#include "myFunctions.cc"

// useful functions
bool selections(int xnum, double eneCent, double qualX, double qualY, int table, int simul );
bool atBorder(int xtalNum);

// Main program
int main ( int argc, char **argv)
{ 
  // --------------------------------------------
  // input parameters
  char inputFileName[500], outputFileName[150];
  int beamEne, appendFlag, simul;
  if (argc==6) {
    strcpy(inputFileName,argv[1]); 
    strcpy(outputFileName,argv[2]);
    beamEne    = atoi(argv[3]);
    appendFlag = atoi(argv[4]);
    simul      = atoi(argv[5]);
  }
  else { 
    cout << " " << endl;
    cout << " Input: 1) input file name  "                           << endl;
    cout << "        2) output file name "                           << endl;
    cout << "        3) beam energy"                                 << endl;
    cout << "        4) appendflag"                                  << endl;
    cout << "        5) simulation (1) or data (0)                 " << endl; 
    cout << endl;
    return 1; 
   }

  // to know what I'm doing:
  cout << endl << endl;
  cout << "transverse shape analysis - running on calibrated rechits!" << endl;
  cout << "beam energy = " << beamEne << endl;
  if (simul == 0){ cout << "run on testbeam data" << endl;  }
  if (simul == 1){ cout << "run on simulated data" << endl; }
  cout << endl << endl;
  




  // -----------------------------------------------------------------------------------------
  //
  //              INITIALIZATIONS    
  //
  // -----------------------------------------------------------------------------------------
  
  
  // --------------------------------------
  // output files
  char outputDir[150];  
  strcpy(outputDir,"./");
  char outputFileTxtRatio[150];  
  strcpy(outputFileTxtRatio,outputDir);
  strcat(outputFileTxtRatio,outputFileName);
  strcat(outputFileTxtRatio,"_Ratios.txt");
  char outputFileTxtMatrix[150];  
  strcpy(outputFileTxtMatrix,outputDir);
  strcat(outputFileTxtMatrix,outputFileName);
  strcat(outputFileTxtMatrix,"_Matrix.txt");
  char outputFileMcpTxt[150];
  strcpy(outputFileMcpTxt,outputDir);
  strcat(outputFileMcpTxt,outputFileName);
  strcat(outputFileMcpTxt,"_mcp.txt");
  

  // --------------------------------------
  // variables
  char outfile[200];
  char name[200];
  char title[200], titleXax[200], titleXay[200];
  TFile *file;
  
  // histo range
  double infY = 0.;
  double supY = 200.;
  int nbinY = 100;
  if(beamEne == 20){
    infY = 0.;  supY = 20.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 20 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 30){
    infY = 10.;  supY = 30.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 30 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 40){
    infY = 10.;  supY = 50.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 40 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 50){
    infY = 10.;  supY = 50.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 50 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 80){
    infY = 20.;  supY = 80.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 80 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 100){
    infY = 20.;  supY = 100.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 100 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 120){
    infY = 30.;  supY = 120.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 120 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 150){
    infY = 40.;  supY = 150.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 150 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  if(beamEne == 500){
    infY = 200.;  supY = 450.;  nbinY = (int)((supY - infY)/0.20);  
    cout << "ene = 150 --> histo range from " << infY << " to " << supY << " with " << nbinY << " bins" << endl;
  }
  

  // ------------------------------------------
  // reading the list of input files with trees and building the chain
  std::vector<int> treeEntries;    // progressive number of entries when adding files
  treeEntries.push_back(0);  
  
  TChain *T = new TChain("T1");
  char Buffer[500];
  char MyRootFile[2000];  // [max length filename]
  cout << "input: " << inputFileName << endl; 
  ifstream *inputFile = new ifstream(inputFileName);
  while( !(inputFile->eof()) ){
    inputFile->getline(Buffer,500);
    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer))){
      sscanf(Buffer,"%s",MyRootFile);
      T->Add(MyRootFile);	  
      treeEntries.push_back( T->GetEntries() );  
      cout << "chaining " << MyRootFile << endl;
    }}
  inputFile->close();
  delete inputFile;


  const static int numTrees = treeEntries.size();
  cout << endl;  
  cout << "in total " << (numTrees-1) << " files have been chained" << endl;
  for (int ii=0; ii< (numTrees-1); ii++){cout << "file " << ii << ", from entry " << treeEntries[ii] << ", to entry " << treeEntries[ii+1] << endl; }
  cout << endl;
  


  // ------------------------------------------
  // Tree construction
  int run, event;
  int xtalSM;
  int xtalEta, xtalPhi;
  int tbMoving;
  int crystal[49];
  double amplit[49];
  double hodoX, hodoY;
  double hodoQualityX, hodoQualityY;
  double hodoSlopeX, hodoSlopeY;
  //
  T->SetMakeClass(1);  
  T->SetBranchStatus("*",0);
  T->SetBranchStatus("run",1);
  T->SetBranchStatus("event",1);
  T->SetBranchStatus("xtalSM",1);
  T->SetBranchStatus("xtalEta",1);
  T->SetBranchStatus("xtalPhi",1);
  T->SetBranchStatus("amplit",1);
  T->SetBranchStatus("tbMoving",1);
  T->SetBranchStatus("hodoX",1);
  T->SetBranchStatus("hodoY",1);
  T->SetBranchStatus("hodoQualityX",1);
  T->SetBranchStatus("hodoQualityY",1);
  T->SetBranchStatus("hodoSlopeX",1);
  T->SetBranchStatus("hodoSlopeY",1);
  T->SetBranchStatus("crystal",1);
  //
  T->SetBranchAddress("run",&run);
  T->SetBranchAddress("event",&event);
  T->SetBranchAddress("xtalSM",&xtalSM);
  T->SetBranchAddress("xtalEta",&xtalEta);
  T->SetBranchAddress("xtalPhi",&xtalPhi);
  T->SetBranchAddress("amplit",amplit);
  T->SetBranchAddress("tbMoving",&tbMoving);
  T->SetBranchAddress("hodoX",&hodoX);
  T->SetBranchAddress("hodoY",&hodoY);
  T->SetBranchAddress("hodoQualityX",&hodoQualityX);
  T->SetBranchAddress("hodoQualityY",&hodoQualityY);
  T->SetBranchAddress("hodoSlopeX",&hodoSlopeX);
  T->SetBranchAddress("hodoSlopeY",&hodoSlopeY);
  T->SetBranchAddress("crystal",crystal);
  
  // entries
  float nEnt=T->GetEntries();
  cout << "Total number of events in loop is " << nEnt << endl; 
  




  // -----------------------------------------------------------------------------------------
  //
  //  FIRST LOOP ON EVENTS: registering the number of crystals             
  //
  // -----------------------------------------------------------------------------------------
  
  // first loop on the events: registering the number of crystals 
  int thisXinBeam = 0;
  int prevXinBeam = 0;
  std::vector<int> xInBeam, eInBeam;
  for (int entry0=0; entry0<nEnt; entry0++) { 
    
    T -> GetEntry(entry0);
    
    // registering the first crystal
    if (entry0 == 0){ 
      xInBeam.push_back(xtalSM); 
      eInBeam.push_back(xtalEta); 
    }        
    
    // did the crystal change?
    prevXinBeam = thisXinBeam;
    if ( (entry0 == 0) || (prevXinBeam == 0) ){ prevXinBeam = xtalSM; }
    thisXinBeam = xtalSM;
    if ( thisXinBeam != prevXinBeam ){ 
      cout << "entry " << entry0 << ", event = " << event 
	   << ": crystal changed--> before " << prevXinBeam << ", now " << thisXinBeam << endl; } 
    
    // checking we are not coming back
    if ( thisXinBeam != prevXinBeam ){ 
      bool itisnew = true; 
      for(std::vector<int>::const_iterator myXib = xInBeam.begin(); myXib != xInBeam.end(); ++myXib){
	if (thisXinBeam == *myXib){ itisnew = false; }
      }	  
      if ( itisnew){ xInBeam.push_back(thisXinBeam); eInBeam.push_back(xtalEta); }  
    }
  }
  
  // number of crystals in beam in the whole scan and corresponding eta
  int xibC = 0;
  int eibC = 0;
  const static int numXinB = xInBeam.size();
  const static int numEinB = eInBeam.size();
  int myXinB[numXinB];
  int myEinB[numXinB];
  for(std::vector<int>::const_iterator myXib = xInBeam.begin(); myXib != xInBeam.end(); ++myXib){
    myXinB[xibC] = *myXib;
    xibC++;
  }
  for(std::vector<int>::const_iterator myEib = eInBeam.begin(); myEib != eInBeam.end(); ++myEib){
    myEinB[eibC] = *myEib;
    eibC++;
  }
  
  
  // to summarize
  cout << endl; 
  cout << "in total: " << xInBeam.size() << " crystals" << endl;
  cout << "Xtals in beam:" << endl;
  for (int myXib=0; myXib<numXinB; myXib++){ 
    cout << "number " << myXib << ": xtal " << myXinB[myXib] << ", eta = " << myEinB[myXib] << endl; }
  cout << endl;
  cout << endl;
  cout << "NOW start the analysis" << endl;
  cout << endl;
  cout << endl;
  
  

  // variables for different xtals  -----------------------------
  int totalInTheRun = 0;
  int wrongXtalInTheRun = 0;
  int wrongQuality[numXinB];
  int movingTable[numXinB];       
  int highAmpl[numXinB]; 
  int negaAmpl[numXinB]; 
  int good0[numXinB];     
  int good1[numXinB];     
  int good2[numXinB];     
  int good3[numXinB];
  int wrongMaxAmp[numXinB];
  for(int myXib=0; myXib<numXinB; myXib++){
    movingTable[myXib]  = 0;
    wrongQuality[myXib] = 0;
    highAmpl[myXib]     = 0;
    negaAmpl[myXib]     = 0;
    good0[myXib]        = 0;
    good1[myXib]        = 0;
    good2[myXib]        = 0;
    good3[myXib]        = 0;
    wrongMaxAmp[myXib]  = 0;
  }
  
  
  
  
 
  // -----------------------------------------------------------------------------------------
  //
  //  SECOND LOOP ON EVENTS: checking everything is fine
  //
  // -----------------------------------------------------------------------------------------
  TH2F *H_ampVsX_bef[numXinB];     
  TH2F *H_ampVsX_aft[numXinB];     
  TH2F *H_ampVsY_bef[numXinB];     
  TH2F *H_ampVsY_aft[numXinB];     
  TH2F *H_MovVsEve_bef[numXinB];
  TH2F *H_MovVsEve_aft[numXinB];
  TH1F *H_hodoX_bef[numXinB];     
  TH1F *H_hodoX_aft[numXinB];     
  TH1F *H_hodoY_bef[numXinB];     
  TH1F *H_hodoY_aft[numXinB];     
  TH1F *H_hodoQualityX_bef[numXinB];     
  TH1F *H_hodoQualityY_bef[numXinB];     
  
  for (int myXib=0; myXib<numXinB; myXib++){
    
    sprintf (name,  "H_ampVsX_bef[%d]", myXib);
    sprintf (title, "E1 vs X - before cuts");    
    H_ampVsX_bef[myXib] = new TH2F(name, title, 80, -20., 20., 150, 0., 150.); 
    
    sprintf (name,  "H_ampVsX_aft[%d]", myXib);
    sprintf (title, "E1 vs X - after cuts");    
    H_ampVsX_aft[myXib] = new TH2F(name, title, 80, -20., 20., 150, 0., 150.); 
    
    sprintf (name,  "H_ampVsY_bef[%d]", myXib);
    sprintf (title, "E1 vs Y - before cuts");    
    H_ampVsY_bef[myXib] = new TH2F(name, title, 80, -20., 20., 150, 0., 150.); 
    
    sprintf (name,  "H_ampVsY_aft[%d]", myXib);
    sprintf (title, "E1 vs Y - after cuts");    
    H_ampVsY_aft[myXib] = new TH2F(name, title, 80, -20., 20., 150, 0., 150.); 
    
    sprintf (name,  "H_MovVsEve_bef[%d]", myXib);
    sprintf (title, "Table checking, before cuts");    
    H_MovVsEve_bef[myXib] = new TH2F(name, title, 80000, 0., 800000., 2, 0., 2.);
    
    sprintf (name,  "H_MovVsEve_aft[%d]", myXib);
    sprintf (title, "Table checking, after cuts");    
    H_MovVsEve_aft[myXib] = new TH2F(name, title, 80000, 0., 800000., 2, 0., 2.);
    
    sprintf (name,  "H_hodoX_bef[%d]", myXib);
    sprintf (title, "HodoX, before cuts");    
    H_hodoX_bef[myXib] = new TH1F(name, title, 80, -20., 20.);
    
    sprintf (name,  "H_hodoX_aft[%d]", myXib);
    sprintf (title, "HodoX, after cuts");    
    H_hodoX_aft[myXib] = new TH1F(name, title, 80, -20., 20.);
    
    sprintf (name,  "H_hodoY_bef[%d]", myXib);
    sprintf (title, "HodoY, before cuts");    
    H_hodoY_bef[myXib] = new TH1F(name, title, 80, -20., 20.);
    
    sprintf (name,  "H_hodoY_aft[%d]", myXib);
    sprintf (title, "HodoY, after cuts");    
    H_hodoY_aft[myXib] = new TH1F(name, title, 80, -20., 20.);
    
    sprintf (name,  "H_hodoQualityX_bef[%d]", myXib);
    sprintf (title, "Hodo quality X, before cuts");    
    H_hodoQualityX_bef[myXib] = new TH1F(name, title, 100, 0., 6.);
    
    sprintf (name,  "H_hodoQualityY_bef[%d]", myXib);
    sprintf (title, "Hodo quality Y, before cuts");    
    H_hodoQualityY_bef[myXib] = new TH1F(name, title, 100, 0., 6.);  
  }  
  
  
  

  // ----------------------------------------
  for (int entry1=0; entry1<nEnt; entry1++) { 
    
    // charging the value in the branches
    T -> GetEntry(entry1);
    if (entry1%100000 == 0){ cout << "Check loop: entry = " << entry1 << endl; }
    
    // table movement
    int table = tbMoving;
    
    // xtal in beam
    int xtalInBeam = xtalSM;
    int myHere = 2000;
    for (int myXib=0; myXib<numXinB; myXib++){ if (xtalInBeam == myXinB[myXib]){myHere = myXib;} }
    
    // to know which file I'm analizing
    int thisII = 0;
    for (int ii=0; ii< (numTrees-1); ii++){if ( (entry1 >= treeEntries[ii]) && (entry1 < treeEntries[ii+1]) ){ thisII = ii; } }
    int globalEve = event + treeEntries[thisII];   // number of the event in the chain
    
    // histos before selections
    H_MovVsEve_bef[myHere]     -> Fill(globalEve, table); 
    H_ampVsX_bef[myHere]       -> Fill(hodoX, amplit[24]);
    H_ampVsY_bef[myHere]       -> Fill(hodoY, amplit[24]);
    H_hodoX_bef[myHere]        -> Fill(hodoX);     
    H_hodoY_bef[myHere]        -> Fill(hodoY);
    H_hodoQualityX_bef[myHere] -> Fill(hodoQualityX);
    H_hodoQualityY_bef[myHere] -> Fill(hodoQualityY);
    
    // selections (no cut on max amplit xtal here) 
    bool select = selections(myHere, amplit[24], hodoQualityX, hodoQualityY, table, simul);
    if (!select){ continue; }
    good0[myHere]++;       
    
    // histos after selections
    H_MovVsEve_aft[myHere]     -> Fill(globalEve, table); 
    H_ampVsX_aft[myHere]       -> Fill(hodoX, amplit[24]);
    H_ampVsY_aft[myHere]       -> Fill(hodoY, amplit[24]);
    H_hodoX_aft[myHere]        -> Fill(hodoX);     
    H_hodoY_aft[myHere]        -> Fill(hodoY);
  }
  
  
  // saving histos
  strcpy(outfile,outputDir);
  strcat(outfile,outputFileName);
  strcat(outfile,"_checks.root");    
  if(appendFlag==0){ file = new TFile (outfile,"RECREATE","istogrammi"); }
  else             { file = new TFile (outfile,"UPDATE","istogrammi");   }
  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    
    sprintf (name, "H_ampVsX_bef_x%d", myXinB[myXib]); 
    H_ampVsX_bef[myXib]->SetTitle(name);
    H_ampVsX_bef[myXib]->Write(name);
    
    sprintf (name, "H_ampVsY_bef_x%d", myXinB[myXib]); 
    H_ampVsY_bef[myXib]->SetTitle(name);
    H_ampVsY_bef[myXib]->Write(name);
    
    sprintf (name, "H_ampVsX_aft_x%d", myXinB[myXib]); 
    H_ampVsX_aft[myXib]->SetTitle(name);
    H_ampVsX_aft[myXib]->Write(name);
    
    sprintf (name, "H_ampVsY_aft_x%d", myXinB[myXib]);
    H_ampVsY_aft[myXib]->SetTitle(name);
    H_ampVsY_aft[myXib]->Write(name);
    
    sprintf (name, "H_MovVsEve_bef_x%d", myXinB[myXib]); 
    H_MovVsEve_bef[myXib]->SetTitle(name);
    H_MovVsEve_bef[myXib]->Write(name);
    
    sprintf (name, "H_MovVsEve_aft_x%d", myXinB[myXib]); 
    H_MovVsEve_aft[myXib]->SetTitle(name);
    H_MovVsEve_aft[myXib]->Write(name);
    
    sprintf (name, "H_hodoX_bef_x%d", myXinB[myXib]); 
    H_hodoX_bef[myXib]->SetTitle(name);
    H_hodoX_bef[myXib]->Write(name);
    
    sprintf (name, "H_hodoX_aft_x%d", myXinB[myXib]); 
    H_hodoX_aft[myXib]->SetTitle(name);
    H_hodoX_aft[myXib]->Write(name);
    
    sprintf (name, "H_hodoY_bef_x%d", myXinB[myXib]); 
    H_hodoY_bef[myXib]->SetTitle(name);
    H_hodoY_bef[myXib]->Write(name);
    
    sprintf (name, "H_hodoY_aft_x%d", myXinB[myXib]); 
    H_hodoY_aft[myXib]->SetTitle(name);
    H_hodoY_aft[myXib]->Write(name);
    
    sprintf (name, "H_hodoQualityX_bef_x%d", myXinB[myXib]); 
    H_hodoQualityX_bef[myXib]->SetTitle(name);
    H_hodoQualityX_bef[myXib]->Write(name);
    
    sprintf (name, "H_hodoQualityY_bef_x%d", myXinB[myXib]); 
    H_hodoQualityY_bef[myXib]->SetTitle(name);
    H_hodoQualityY_bef[myXib]->Write(name);
  }
  
  // deleting
  for (int myXib=0; myXib<numXinB; myXib++){ 
    delete H_ampVsX_bef[myXib];     
    delete H_ampVsX_aft[myXib];     
    delete H_ampVsY_bef[myXib];    
    delete H_ampVsY_aft[myXib];     
    delete H_MovVsEve_bef[myXib];
    delete H_MovVsEve_aft[myXib];
    delete H_hodoX_bef[myXib];     
    delete H_hodoX_aft[myXib];     
    delete H_hodoY_bef[myXib];     
    delete H_hodoY_aft[myXib];     
    delete H_hodoQualityX_bef[myXib];     
    delete H_hodoQualityY_bef[myXib];     
  }
  
  file->Close();
  delete file;
  
  



  // -----------------------------------------------------------------------------------------
  //
  //  THIRD LOOP ON EVENTS: first search loop for maximum containment point
  //
  // -----------------------------------------------------------------------------------------

  // region definition: CUT on the impinging particles position       ------------------
  double UpxFitX   =  10.;       // limits for fit X       
  double DownxFitX = -10.;
  double UpyFitX   =   4.;
  double DownyFitX =  -4.;
  double UpxFitY   =   4.;       // limits for fit Y       
  double DownxFitY =  -4.;
  double UpyFitY   =  10.;
  double DownyFitY = -10.;

  // preparing histos for each crystal  
  TH2F *H_mapXhodoVsE_1[numXinB];     
  TH2F *H_mapYhodoVsE_1[numXinB];     
  for (int myXib=0; myXib<numXinB; myXib++){
    sprintf (name,  "H_mapXhodoVsE_1[%d]", myXib);
    sprintf (title, "Pulse maximum vs X");    
    H_mapXhodoVsE_1[myXib] = new TH2F(name, title, 80, -20., 20., nbinY, infY, supY);
    H_mapXhodoVsE_1[myXib] -> GetXaxis()->SetTitle("X (mm)");
    H_mapXhodoVsE_1[myXib] -> GetYaxis()->SetTitle("amplit (adc counts)");
    
    sprintf (name,  "H_mapYhodoVsE_1[%d]", myXib);
    sprintf (title, "Pulse maximum vs Y");    
    H_mapYhodoVsE_1[myXib] = new TH2F(name, title, 80, -20., 20., nbinY, infY, supY);
    H_mapYhodoVsE_1[myXib] -> GetXaxis()->SetTitle("Y (mm)");
    H_mapYhodoVsE_1[myXib] -> GetYaxis()->SetTitle("amplit (adc counts)");
   }
  
  
  // analysis --------------- 
  for (int entry2=0; entry2<nEnt; entry2++) { 
    
    // counting the events
    totalInTheRun++; 
     
    // charging the value in the branches
    T -> GetEntry(entry2);
    if (entry2%100000 == 0){ cout << "First analysis loop: entry = " << entry2 << endl; }
    
    // table movement
    int table = tbMoving;
    
    // xtal in beam
    int xtalInBeam = xtalSM;
    int myHere = 2000;
    for (int myXib=0; myXib<numXinB; myXib++){ if (xtalInBeam == myXinB[myXib]){myHere = myXib;} }
    
    // to know which file I'm analizing
    int thisII = 0;
    for (int ii=0; ii< (numTrees-1); ii++){ if ( (entry2 >= treeEntries[ii]) && (entry2 < treeEntries[ii+1]) ){ thisII = ii; } }
    
    // selections (no cut on max amplit xtal here)  
    bool select = true;
    if ( myHere >1700)       { wrongXtalInTheRun++; select = false; }
    if ( amplit[24] > 999.)  { highAmpl[myHere]++;  select = false; }
    if ( amplit[24] < 0.5)   { negaAmpl[myHere]++;  select = false; }
    if ( !simul)             { 
      if (table)                                    { movingTable[myHere]++;  select = false; }
      if ((hodoQualityX>2.0) || (hodoQualityX<0.0)) { wrongQuality[myHere]++; select = false; }
      if ((hodoQualityY>2.0) || (hodoQualityY<0.0)) { wrongQuality[myHere]++; select = false; }
    }
    if (!select){ continue; }
    good1[myHere]++;       
    
    // basic cuts on the position
    if ((hodoX < DownxFitX) || (hodoX > UpxFitX)){continue;}
    if ((hodoY < DownyFitY) || (hodoY > UpyFitY)){continue;}
    
    // maximum containment point: filling histos, after cuts on the impact point
    if ((DownyFitX<hodoY) && (hodoY<UpyFitX)){ H_mapXhodoVsE_1[myHere] -> Fill(hodoX, amplit[24]); }
    if ((DownxFitY<hodoX) && (hodoX<UpxFitY)){ H_mapYhodoVsE_1[myHere] -> Fill(hodoY, amplit[24]); }
    
  } // end loop over entries
  

  // First fit with the polynomial function + save histos
  double p0x_1[numXinB], p0y_1[numXinB];
  for (int myXib=0; myXib<numXinB; myXib++){ 
    p0x_1[myXib] = 1000.;
    p0y_1[myXib] = 1000.;
  }   
  
  strcpy(outfile,outputDir);
  strcat(outfile,outputFileName);
  strcat(outfile,"_histos.root");    
  if(appendFlag==0){ file = new TFile (outfile,"RECREATE","istogrammi"); }
  else             { file = new TFile (outfile,"UPDATE","istogrammi");   }
  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    
    // fitting
    p0x_1[myXib] = Fit_MaximumPoint( H_mapXhodoVsE_1[myXib], 10);
    p0y_1[myXib] = Fit_MaximumPoint( H_mapYhodoVsE_1[myXib], 10);

    // saving
    sprintf (name, "H_mapXhodoVsE_1loop_x%d", myXinB[myXib]); 
    H_mapXhodoVsE_1[myXib]->SetTitle(name);
    H_mapXhodoVsE_1[myXib]->Write(name);

    sprintf (name, "H_mapYhodoVsE_1loop_x%d", myXinB[myXib]); 
    H_mapYhodoVsE_1[myXib]->SetTitle(name);
    H_mapYhodoVsE_1[myXib]->Write(name);
  }
  file->Close();
  delete file;
  
  
  // deleting
  for (int myXib=0; myXib<numXinB; myXib++){ 
    delete H_mapXhodoVsE_1[myXib];     
    delete H_mapYhodoVsE_1[myXib];     
  }
  
  
  
  // -----------------------------------------------------------------------------------------
  //
  //  FOURTH LOOP ON EVENTS: second search loop for maximum containment point
  //
  // -----------------------------------------------------------------------------------------

  // New limits ------------------------------------
  double DownyFitX_v[numXinB];  
  double DownxFitY_v[numXinB];  
  double UpyFitX_v[numXinB];  
  double UpxFitY_v[numXinB];  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    DownyFitX_v[myXib] = p0y_1[myXib]-2;
    DownxFitY_v[myXib] = p0x_1[myXib]-2;
    UpyFitX_v[myXib]   = p0y_1[myXib]+2;
    UpxFitY_v[myXib]   = p0x_1[myXib]+2;
  }
  
  // preparing histos for each crystal  
  TH2F *H_mapXhodoVsE_2[numXinB];     
  TH2F *H_mapYhodoVsE_2[numXinB];     
  for (int myXib=0; myXib<numXinB; myXib++){
    
    sprintf (name,  "H_mapXhodoVsE_2[%d]", myXib);
    sprintf (title, "Pulse maximum vs X");    
    H_mapXhodoVsE_2[myXib] = new TH2F(name, title, 80, -20., 20., nbinY, infY, supY);
    H_mapXhodoVsE_2[myXib] -> GetXaxis()->SetTitle("X (mm)");
    H_mapXhodoVsE_2[myXib] -> GetYaxis()->SetTitle("amplit (adc counts)");
    
    sprintf (name,  "H_mapYhodoVsE_2[%d]", myXib);
    sprintf (title, "Pulse maximum vs Y");    
    H_mapYhodoVsE_2[myXib] = new TH2F(name, title, 80, -20., 20., nbinY, infY, supY);
    H_mapYhodoVsE_2[myXib] -> GetXaxis()->SetTitle("Y (mm)");
    H_mapYhodoVsE_2[myXib] -> GetYaxis()->SetTitle("amplit (adc counts)");
  }
  
  for (int entry3=0; entry3<nEnt; entry3++) { 
    
    // charging the value in the branches
    T -> GetEntry(entry3);
    if (entry3%100000 == 0){ cout << "Second analysis loop: entry = " << entry3 << endl; }
    
    // xtal in beam
    int xtalInBeam = xtalSM;
    int myHere = 2000;
    for (int myXib=0; myXib<numXinB; myXib++){ if (xtalInBeam == myXinB[myXib]){myHere = myXib;} }
    
    // table movement
    int table = tbMoving;
    
    // to know which file I'm analizing
    int thisII = 0;
    for (int ii=0; ii< (numTrees-1); ii++){if ( (entry3 >= treeEntries[ii]) && (entry3 < treeEntries[ii+1]) ){ thisII = ii; } }
    
    // selections (no cut on max amplit xtal here) 
    bool select = selections(myHere, amplit[24], hodoQualityX, hodoQualityY, table, simul); 
    if (!select){ continue; }
    good2[myHere]++;       
    
    // basic cuts on the position
    if ((hodoX < DownxFitX) || (hodoX > UpxFitX)){continue;}
    if ((hodoY < DownyFitY) || (hodoY > UpyFitY)){continue;}

    // maximum containment point: filling histos, after cuts on the impact point
    if ((DownyFitX_v[myHere]<hodoY) && (hodoY<UpyFitX_v[myHere])){ H_mapXhodoVsE_2[myHere] -> Fill(hodoX, amplit[24]); }
    if ((DownxFitY_v[myHere]<hodoX) && (hodoX<UpxFitY_v[myHere])){ H_mapYhodoVsE_2[myHere] -> Fill(hodoY, amplit[24]); }
     
  } // end loop over entries
  

  
  // ---------------------------------------------------------
  // Second fit with the polynomial function 
  strcpy(outfile,outputDir);
  strcat(outfile,outputFileName);
  strcat(outfile,"_histos.root");    
  file = new TFile (outfile,"UPDATE","istogrammi");   
  
  
  double p0x_2[numXinB], p0y_2[numXinB];
  double p0x_c22[numXinB],  p0y_c22[numXinB];
  Stat_t p0x_ent2[numXinB], p0y_ent2[numXinB];
  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    
    // initializing
    p0x_2[myXib]    =  1000.;
    p0y_2[myXib]    =  1000.;
    p0x_c22[myXib]  =  1000.;
    p0y_c22[myXib]  =  1000.;
    p0x_ent2[myXib] = -15000;
    p0y_ent2[myXib] = -15000;
    
    // fitting: value
    p0x_2[myXib] = Fit_MaximumPoint( H_mapXhodoVsE_2[myXib], 10);
    p0y_2[myXib] = Fit_MaximumPoint( H_mapYhodoVsE_2[myXib], 10);
    
    // fitting: chi2s
    p0x_c22[myXib]  = Fit_MaximumPoint( H_mapXhodoVsE_2[myXib], 30);
    p0y_c22[myXib]  = Fit_MaximumPoint( H_mapYhodoVsE_2[myXib], 30);
    
    // histo entries
    p0x_ent2[myXib] = H_mapXhodoVsE_2[myXib]->GetEntries();
    p0y_ent2[myXib] = H_mapYhodoVsE_2[myXib]->GetEntries();

    // saving
    sprintf (name, "H_mapXhodoVsE_2loop_x%d", myXinB[myXib]); 
    H_mapXhodoVsE_2[myXib]->SetTitle(name);
    H_mapXhodoVsE_2[myXib]->Write(name);

    sprintf (name, "H_mapYhodoVsE_2loop_x%d", myXinB[myXib]); 
    H_mapYhodoVsE_2[myXib]->SetTitle(name);
    H_mapYhodoVsE_2[myXib]->Write(name);
  }
  file->Close();
  delete file;
  
  // deleting
  for (int myXib=0; myXib<numXinB; myXib++){ 
    delete H_mapXhodoVsE_2[myXib];     
    delete H_mapYhodoVsE_2[myXib];     
  }
  
  
  
  
  // --------------------------------------------------------------
  // now we have the maximum containment point for each xtal 
  ofstream *ResFileMc;
  if(appendFlag==0){
    ResFileMc = new ofstream(outputFileMcpTxt,ios::out);
    *ResFileMc << "#Xtal"    << "\t"
	       << "entriesX" << "\t" << "maxX" << "\t" << "chi2X" << "\t"
	       << "entriesY" << "\t" << "maxY" << "\t" << "chi2Y" << endl; }
  else { ResFileMc = new ofstream(outputFileMcpTxt,ios::app); }
  for (int myXib=0; myXib<numXinB; myXib++){
    *ResFileMc << myXinB[myXib]   << "\t"  
	       << p0x_ent2[myXib] << "\t" << p0x_2[myXib] << "\t" << p0x_c22[myXib]  << "\t"
	       << p0y_ent2[myXib] << "\t" << p0y_2[myXib] << "\t" << p0y_c22[myXib]  << endl;
  }
  cout << endl;
  

  // to skip xtals where the mcp is not known
  ofstream *badMcPFile = new ofstream("badMcpFile.txt",ios::out);
  map<int, bool> badMcp;
  for (int myXib=0; myXib<numXinB; myXib++){
    bool isItOk = true;
    if ( (p0x_c22[myXib] > 3.5) || (p0y_c22[myXib] > 3.5) ){ isItOk = false; }
    badMcp.insert(pair<int, bool>(myXinB[myXib], isItOk));
  }
  
  


  // -----------------------------------------------------------------------------------------
  //
  //  FIFTH LOOP ON EVENTS: final analysis
  //
  // -----------------------------------------------------------------------------------------
  
  // New limits ------------------------------------
  double downy[numXinB];  
  double downx[numXinB];  
  double upy[numXinB];  
  double upx[numXinB];  
  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    downy[myXib] = p0y_2[myXib]-2;
    downx[myXib] = p0x_2[myXib]-2;
    upy[myXib]   = p0y_2[myXib]+2;
    upx[myXib]   = p0x_2[myXib]+2;
  }
  
  // histos --------------------------------
  TH1F *H_eMatrix[3][numXinB];         // [0] = 1x1         [1] = 3x3         [2] = 5x5    
  TH1F *H_eRatio[3][numXinB];          // [0] = e1/e9       [1] = e1/e25      [2] = e9/e25

  TH1F *H_energyNxN[3];
  TProfile *H_energyieta;
  TProfile *H_energyiphi;

  H_energyieta = new TProfile("H_energyieta","energy by ieta", 5, -2.5, 2.5,0.0,1.0);
  H_energyiphi = new TProfile("H_energyiphi","energy by iphi", 5, -2.5, 2.5,0.0,1.0);

  for (int ii=0; ii<3; ii++){     
    sprintf(name, "H_energy%dx%d",2*ii+1,2*ii+1);
    sprintf(title, "%dx%d crystal energy for beam energy = %d",2*ii+1,2*ii+1,beamEne);
    sprintf(titleXax, "E%dx%d (GeV)",2*ii+1,2*ii+1);
    
    H_energyNxN[ii] = new TH1F(name, title, 280, 0.5, 1.1);
    H_energyNxN[ii]-> GetXaxis()->SetTitle(titleXax);
  }
  TH1F* H_energyE1E9 = new TH1F("H_energyE1E9","Energy ratio of 1x1 to 3x3 crystals;E_{1}/E_{9}",200,0.7,0.9);
  TH1F* H_energyE1E25 = new TH1F("H_energyE1E25","Energy ratio of 1x1 to 5x5 crystals;E_{1}/E_{25}",200,0.7,0.9);
  TH1F* H_energyE9E25 = new TH1F("H_energyE9E25","Energy ratio of 3x3 to 5x5 crystals;E_{9}/E_{25}",200,0.9,1.0);
  
  for (int myXib=0; myXib<numXinB; myXib++){
    for (int ii=0; ii<3; ii++){     
      sprintf (name, "H_eMatrix[%d][%d]", ii, myXib);
      if(ii==0){ sprintf(title, "1x1 energy");  sprintf(titleXax, "E1 (GeV)"); }  
      if(ii==1){ sprintf(title, "3x3 energy");  sprintf(titleXax, "E9 (GeV)"); }   
      if(ii==2){ sprintf(title, "5x5 energy");  sprintf(titleXax, "E25 (GeV)"); }   

      if (beamEne==20){ 
	if (!simul){
	  if (ii==0)         { H_eMatrix[ii][myXib] = new TH1F(name, title,  100,  0., 30.); }
	  if (ii==1 || ii==2){ H_eMatrix[ii][myXib] = new TH1F(name, title,  100,  0., 30.); }
	}
	if (simul){
	  if (ii==0)         { H_eMatrix[ii][myXib] = new TH1F(name, title,  100,  10., 20.); }
	  if (ii==1)         { H_eMatrix[ii][myXib] = new TH1F(name, title,  100,  15., 22.); }
	  if (ii==2)         { H_eMatrix[ii][myXib] = new TH1F(name, title,  100,  15., 22.); }
	}
      }
      else if (beamEne==30) { 
	if (!simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 40.); } 
	  if (ii==1 || ii==2) { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 40.); } 
	}
	if (simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 15., 27.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 21., 32.); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 24., 35.); }  
	}
      }
      else if (beamEne==40) { 
	if (!simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 60.); }  
	  if (ii==1 || ii==2) { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 60.); }  
	}
	if (simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 25., 36.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 31., 42.); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 33., 44.); }  
	}
      }
      else if (beamEne==50) { 
	if (!simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 60.); }  
	  if (ii==1 || ii==2) { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 60.); }  
	}
	if (simul){
	  // electrons
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 26., 43.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 42., 50.); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 45., 52.); }  
	  // pions
	  // if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 43.); }  
	  // if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 50.); }  
	  // if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 52.); }  
	}
      }
      else if (beamEne==80) { 
	if (!simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 90.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 90.); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 0., 90.); }  
	}								 
	if (simul){				          	       					 		 
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 50., 70.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 65., 85.); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title,  100, 68., 88.); }  
	}
      }
      else if (beamEne==100){ 
	if (simul){									    		 
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 60,  60., 90.); } 
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 80., 105.); } 
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 90., 110.); } 
	}
      }
      else if (beamEne==120){ 
	if (!simul){
	  // if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100,  95., 110.); }  
	  // if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 118., 128.); }   
	  // if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 123., 133.); }  
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 85., 100.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 90., 120.); }   
	  // if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 0.95, 1.05); }  
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 100., 130.); }  
	}						   	 			 
	if (simul){									    		 
	  // electrons
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100,  85., 110.); } 
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 106., 126.); } 
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 110., 130.); } 
	  // pions
	  // if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 0., 105.); } 
	  // if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 0., 116.); } 
	  // if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 0., 120.); } 
	}
      }
      else if (beamEne==150){ 
	if (!simul){
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 110., 130.); }  
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 130., 160.); }   
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 140., 160.); }  
	}						   	 			 
	if (simul){									    		 
	  // electrons
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 110., 130.); } 
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 130., 160.); } 
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 140., 160.); } 
	}
      }
      else if (beamEne==500){ 
	if (simul){									    		 
	  //if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 380., 440.); } 
	  //if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 400., 510.); } 
	  //if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 400., 520.); } 
	  //
	  if (ii==0)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 100, 380., 440.); } 
	  if (ii==1)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 101, 440., 500.); } 
	  if (ii==2)          { H_eMatrix[ii][myXib] = new TH1F(name, title, 101, 440., 520.); } 
	}
      }
      H_eMatrix[ii][myXib] -> GetXaxis()->SetTitle(titleXax);
    }

    for (int ii=0; ii<3; ii++){      
      sprintf (name, "H_eRatio[%d][%d]", ii, myXib);
      if(ii==0){ sprintf(title, "E1/E9");   sprintf(titleXax, "E1/E9");  H_eRatio[ii][myXib] = new TH1F(name, title, 80, 0.80, 0.88); }
      if(ii==1){ sprintf(title, "E1/E25");  sprintf(titleXax, "E1/E25"); H_eRatio[ii][myXib] = new TH1F(name, title, 80, 0.75, 0.85);}
      if(ii==2){ sprintf(title, "E9/E25");  sprintf(titleXax, "E9/E25"); H_eRatio[ii][myXib] = new TH1F(name, title, 60, 0.94, 0.98);}
      H_eRatio[ii][myXib] -> GetXaxis()->SetTitle(titleXax);
    }
  }
  


  // -----------------------------------------------

  for (int entry3=0; entry3<nEnt; entry3++) { 
    
    // charging the value in the branches
    T -> GetEntry(entry3);
    if (entry3%100000 == 0){ cout << "Final analysis loop: entry = " << entry3 << endl; }

    // xtal in beam
    int xtalInBeam = xtalSM;
    int myHere = 2000;
    for (int myXib=0; myXib<numXinB; myXib++){ if (xtalInBeam == myXinB[myXib]){myHere = myXib;} }
    
    // table movement
    int table = tbMoving;
    
    // to know which file I'm analizing
    int thisII = 0;
    for (int ii=0; ii< (numTrees-1); ii++){
      if ( (entry3 >= treeEntries[ii]) && (entry3 < treeEntries[ii+1]) ){ thisII = ii; } 
    }

    // basic cuts (no cut on max amplit xtal here) 
    bool select = selections(myHere, amplit[24], hodoQualityX, hodoQualityY, table, simul); 
    if (!select){ continue; }
    good3[myHere]++;      

    // no xtals with bad determination of the max conteinment point
    if(!simul){
      bool mcpFound = true;
      map<int, bool>::const_iterator mcpIt;
      mcpIt = badMcp.find(xtalInBeam);
      if (mcpIt != badMcp.end()){ mcpFound = mcpIt->second; }
      if (!mcpFound){ *badMcPFile << mcpIt->first << endl; continue; }
    }

    // selections: cuts on the impact point 
    if ((hodoX < downx[myHere]) || (hodoX > upx[myHere])){continue;}
    if ((hodoY < downy[myHere]) || (hodoY > upy[myHere])){continue;}

    // no xtals at borders, basic cut to remove what enter at least 3x3
    bool amIatBorder = atBorder(xtalInBeam);
    if (amIatBorder){ continue; }      

    // further selection: only if xtal-in-beam = max-amplitude-xtal
    int maxAmpInMtx = maxAmplitInMatrix(amplit);
    if (maxAmpInMtx != 24){ wrongMaxAmp[myHere]++;  continue; }

    // computing variables
    double e1e9  = -50.;
    double e1e25 = -50.;
    double e9e25 = -50.;
    double e1    = ene1x1_xtal(amplit,24); 
    double e9    = ene3x3_xtal(amplit,24); 
    double e25   = ene5x5_xtal(amplit,24); 

    if( (e1>-20) && (e9>-20) ) { e1e9  = e1/e9;  }
    if( (e1>-20) && (e25>-20) ){ e1e25 = e1/e25; }
    if( (e9>-20) && (e25>-20) ){ e9e25 = e9/e25; }
    
    // filling histos 
    H_eMatrix[0][myHere] -> Fill(e1);
    H_eMatrix[1][myHere] -> Fill(e9);
    H_eMatrix[2][myHere] -> Fill(e25);
    H_eRatio[0][myHere]  -> Fill(e1e9);
    H_eRatio[1][myHere]  -> Fill(e1e25);
    H_eRatio[2][myHere]  -> Fill(e9e25);

    //@@@@syjun
    double Eieta[5] = {0};
    double Eiphi[5] = {0};
    if(beamEne!=0) {
      H_energyNxN[0]->Fill(e1/(1.0*beamEne));
      H_energyNxN[1]->Fill(e9/(1.0*beamEne));
      H_energyNxN[2]->Fill(e25/(1.0*beamEne));
      H_energyE1E9->Fill(e1/e9);
      H_energyE1E25->Fill(e1/e25);
      H_energyE9E25->Fill(e9/e25);

      energy_ieta(amplit,Eieta);
      energy_iphi(amplit,Eiphi);

      for(int ii = 0 ; ii < 5 ; ii++) {
	H_energyieta->Fill(-2.0+1.0*ii,Eieta[ii]/(1.0*beamEne));
	H_energyiphi->Fill(-2.0+1.0*ii,Eiphi[ii]/(1.0*beamEne));
      }

    }

  } // end loop on the events


  cout << endl;
  cout << "End Third Analysis Loop" << endl;
  cout << endl;


  // --------------------------------------------------------
  // Saving histos into files before the fit
  strcpy(outfile,outputDir);
  strcat(outfile,outputFileName);
  strcat(outfile,"_befFit.root");
  if(appendFlag==0){ file = new TFile (outfile,"RECREATE","istogrammi"); }
  else             { file = new TFile (outfile,"UPDATE","istogrammi"); }

  for (int ii=0; ii<3; ii++){     
    sprintf(name, "H_energy%dx%d",2*ii+1,2*ii+1);
    H_energyNxN[ii]->SetTitle(name);
    H_energyNxN[ii]->Write(name);
  }
  H_energyE1E9->Write();
  H_energyE1E25->Write();
  H_energyE9E25->Write();
  H_energyieta->Write();  
  H_energyiphi->Write();  

  for (int myXib=0; myXib<numXinB; myXib++){ 

    for (int ii=0; ii<3; ii++){      
      if(ii==0){ sprintf (name, "H_e1_x%d_ene%d",  myXinB[myXib], beamEne); }
      if(ii==1){ sprintf (name, "H_e9_x%d_ene%d",  myXinB[myXib], beamEne); }
      if(ii==2){ sprintf (name, "H_e25_x%d_ene%d", myXinB[myXib], beamEne); }
      H_eMatrix[ii][myXib] -> SetTitle(name);
      H_eMatrix[ii][myXib] -> Write(name);
    }
    
    for (int ii=0; ii<3; ii++){      
      if(ii==0){ sprintf (name, "H_e1e9_x%d_ene%d",  myXinB[myXib], beamEne); }
      if(ii==1){ sprintf (name, "H_e1e25_x%d_ene%d", myXinB[myXib], beamEne); }
      if(ii==2){ sprintf (name, "H_e9e25_x%d_ene%d", myXinB[myXib], beamEne); }
      H_eRatio[ii][myXib] -> SetTitle(name);
      H_eRatio[ii][myXib] -> Write(name);
    }
  }

  file->Close();
  delete file;


  // --------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  //
  //  FITTING 
  //
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------

  // NxN matrices
  ofstream *ResFileMatrix;
  if(appendFlag==0){ 
    ResFileMatrix = new ofstream(outputFileTxtMatrix,ios::out);    
    *ResFileMatrix << "#Energy"  << "\t" << "xtal"        << "\t" << "eta" << "\t"  
		   << "sample"  << "\t" << "entries"     << "\t" 
		   << "H_mean"  << "\t" << "H_rms"       << "\t" 
		   << "G_mean"  << "\t" << "G_mean err"  << "\t" 
		   << "G_sigma" << "\t" << "G_sigma err" << endl;} 
  else { ResFileMatrix = new ofstream(outputFileTxtMatrix,ios::app); }  


  // fitting 
  for (int myXib=0; myXib<numXinB; myXib++){ 
        
    for (int ii=0; ii<3; ii++){ 

      // histo parameters
      Stat_t h_entries = H_eMatrix[ii][myXib]->GetEntries();      
      int peakBin      = H_eMatrix[ii][myXib]->GetMaximumBin();
      double h_norm    = H_eMatrix[ii][myXib]->GetMaximum();
      double h_rms     = H_eMatrix[ii][myXib]->GetRMS();
      double h_mean    = H_eMatrix[ii][myXib]->GetMean();
      double h_peak    = H_eMatrix[ii][myXib]->GetBinCenter(peakBin);


      // gaussian fit to initialize
      TF1 *gausa = new TF1 ("gausa","[0]*exp(-1*(x-[1])*(x-[1])/2/[2]/[2])",h_peak-10*h_rms,h_peak+10*h_rms);
      gausa->SetParameters(h_norm,h_peak,h_rms);
      H_eMatrix[ii][myXib]->Fit(gausa,"","",h_peak-3*h_rms,h_peak+3*h_rms);
      double gausNorm  = gausa->GetParameter(0);
      double gausMean  = gausa->GetParameter(1);
      double gausSigma = fabs(gausa->GetParameter(2));
      double gausChi2  = gausa->GetChisquare()/gausa->GetNDF();
      if (gausChi2>100){ gausMean = h_peak; gausSigma = h_rms; }

      // crystalball limits
      double myXmin = gausMean - 3.*gausSigma;
      double myXmax = gausMean + 2.*gausSigma;
      
      // crystalball fit
      TF1 *cb_p = new TF1 ("cb_p",crystalball,myXmin,myXmax, 5) ;
      cb_p->SetParNames ("Mean","Sigma","alpha","n","Norm","Constant");
      cb_p->SetParameter(0, gausMean);
      cb_p->SetParameter(1, gausSigma);
      cb_p->FixParameter(3, 5.);    
      // cb_p->SetParameter(3, 5.);     // solo per caso 500 con BGO e 10% BGO birk        
      // cb_p->FixParameter(3, 3.);     // solo x caso 500 no Birk
      cb_p->SetParameter(4, gausNorm);
      cb_p->SetParLimits(2, 0.1, 5.);
      H_eMatrix[ii][myXib]->Fit(cb_p,"lR","",myXmin,myXmax);

      double matrix_gmean      = cb_p->GetParameter(0);
      double matrix_gsigma     = cb_p->GetParameter(1); 
      double matrix_gmean_err  = cb_p->GetParError(0); 
      double matrix_gsigma_err = cb_p->GetParError(1); 

      delete cb_p;
      delete gausa;

      // writing into the text file
      *ResFileMatrix << beamEne       << "\t" << myXinB[myXib]     << "\t" << myEinB[myXib] << "\t" 
		     << ii            << "\t" << h_entries         << "\t" 
		     << h_mean        << "\t" << h_rms             << "\t" 
		     << matrix_gmean  << "\t" << matrix_gmean_err  << "\t" 
		     << matrix_gsigma << "\t" << matrix_gsigma_err << endl; 
    }
  }
  
  
  // Ratios: 1/9, 1/25, 9/25
  ofstream *ResFileRatio;
  if(appendFlag==0){ 
    ResFileRatio = new ofstream(outputFileTxtRatio,ios::out);    
    *ResFileRatio << "#Ene"    << "\t" << "xtal"        << "\t" << "eta" << "\t" 
		  << "sample"  << "\t" << "entries"     << "\t" 
		  << "H_mean"  << "\t" << "H_rms"       << "\t" 
		  << "G_mean"  << "\t" << "G_mean err"  << "\t" 
		  << "G_sigma" << "\t" << "G_sigma err" << endl;} 
  else { ResFileRatio = new ofstream(outputFileTxtRatio,ios::app); }    


  for (int myXib=0; myXib<numXinB; myXib++){ 
    for (int ii=0; ii<3; ii++){ 
      
      // histo parameters
      Stat_t h_entries = H_eRatio[ii][myXib]->GetEntries();      
      int peakBin      = H_eRatio[ii][myXib]->GetMaximumBin();
      double h_norm    = H_eRatio[ii][myXib]->GetMaximum();
      double h_rms     = H_eRatio[ii][myXib]->GetRMS();
      double h_mean    = H_eRatio[ii][myXib]->GetMean();
      double h_peak    = H_eRatio[ii][myXib]->GetBinCenter(peakBin);
      
      // gaussian fit to initialize
      TF1 *gausa = new TF1 ("gausa","[0]*exp(-1*(x-[1])*(x-[1])/2/[2]/[2])",h_peak-10*h_rms,h_peak+10*h_rms);
      gausa->SetParameters(h_norm,h_peak,h_rms);
      H_eRatio[ii][myXib]->Fit(gausa,"","",h_peak-3*h_rms,h_peak+3*h_rms);
      double gausNorm  = gausa->GetParameter(0);
      double gausMean  = gausa->GetParameter(1);
      double gausSigma = fabs(gausa->GetParameter(2));
      double gausChi2  = gausa->GetChisquare()/gausa->GetNDF();
      if (gausChi2>100){ gausMean = h_peak; gausSigma = h_rms; }

      // crystalball limits
      double myXmin = gausMean - 3.*gausSigma;
      double myXmax = gausMean + 2.*gausSigma;
      
      // crystalball fit
      TF1 *cb_p = new TF1 ("cb_p",crystalball,myXmin,myXmax, 5) ;
      cb_p->SetParNames ("Mean","Sigma","alpha","n","Norm","Constant");
      cb_p->SetParameter(0, gausMean);
      cb_p->SetParameter(1, gausSigma);
      cb_p->FixParameter(3, 5.);
      cb_p->SetParameter(4, gausNorm);
      cb_p->SetParLimits(2, 0.1, 5.);
      H_eRatio[ii][myXib]->Fit(cb_p,"lR","",myXmin,myXmax);

      double ratio_gmean      = cb_p->GetParameter(0);
      double ratio_gsigma     = cb_p->GetParameter(1); 
      double ratio_gmean_err  = cb_p->GetParError(0); 
      double ratio_gsigma_err = cb_p->GetParError(1); 

      delete cb_p;
      delete gausa;


      // writing into the text file
      *ResFileRatio << beamEne      << "\t" << myXinB[myXib]    << "\t" << myEinB[myXib] << "\t" 
 		    << ii           << "\t" << h_entries        << "\t" 
		    << h_mean       << "\t" << h_rms            << "\t" 
		    << ratio_gmean  << "\t" << ratio_gmean_err  << "\t" 
		    << ratio_gsigma << "\t" << ratio_gsigma_err << endl; 
    }
  }
  

  // ---------------------------------------------------------
  // Saving histos into files after the fit
  strcpy(outfile,outputDir);
  strcat(outfile,outputFileName);
  strcat(outfile,"_aftFit.root");
  if(appendFlag==0){ file = new TFile (outfile,"RECREATE","istogrammi"); }
  else             { file = new TFile (outfile,"UPDATE","istogrammi"); }  
  for (int myXib=0; myXib<numXinB; myXib++){ 
    
    for (int ii=0; ii<3; ii++){      
      if(ii==0){ sprintf (name, "H_e1_x[%d]",  myXinB[myXib]); }
      if(ii==1){ sprintf (name, "H_e9_x[%d]",  myXinB[myXib]); }
      if(ii==2){ sprintf (name, "H_e25_x[%d]", myXinB[myXib]); }
      H_eMatrix[ii][myXib] -> Write(name);
    }
    
    for (int ii=0; ii<3; ii++){      
      if(ii==0){ sprintf (name, "H_e1e9_x[%d]",  myXinB[myXib]); }
      if(ii==1){ sprintf (name, "H_e1e25_x[%d]", myXinB[myXib]); }
      if(ii==2){ sprintf (name, "H_e9e25_x[%d]", myXinB[myXib]); }
      H_eRatio[ii][myXib] -> Write(name);
    }    
  }
  file->Close();
  delete file;



  
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  //
  //  CONCLUSIONS
  //
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------------

  // deleting --------------------------
  for (int myXib=0; myXib<numXinB; myXib++){
    for (int ii=0; ii<3; ii++){       
      delete H_eMatrix[ii][myXib];
      delete H_eRatio[ii][myXib];
    }
  }  

  // statistics ----------------------------
  cout << endl;
  cout << endl;
  cout << totalInTheRun        << " total events" << endl;
  cout << wrongXtalInTheRun    << " not existing xtal" << endl;
  cout << endl;
  for (int myXib=0; myXib<numXinB; myXib++){ 
    cout << "xtal " << myXinB[myXib] << endl; 
    cout << movingTable[myXib]  << " moving table" << endl;
    cout << highAmpl[myXib]     << " too high amplitude" << endl;
    cout << negaAmpl[myXib]     << " negative amplitude" << endl;
    cout << wrongQuality[myXib] << " bad hodoscope quality" << endl;
    cout << endl;
    cout << good0[myXib]        << " good events at check loop" << endl;
    cout << good1[myXib]        << " good events at 1st loop"   << endl;
    cout << good2[myXib]        << " good events at 2nd loop"   << endl;
    cout << good3[myXib]        << " good events at 3rd loop"   << endl;
    cout << "after other selections: " << wrongMaxAmp[myXib] << " xib != max amplitude xtal" << endl; 
    cout << endl;
  }

} // end main




// functions

// selections
bool selections(int xnum, double eneCent, double qualX, double qualY, int table, int simul){

  bool select = true;
  if ( xnum > 1700 )                                    { select = false; }        // problems with the xtal-in-beam
  if ( eneCent > 999. )                                 { select = false; }        // problems with amplitudes        
  if ( eneCent < 0.5)                                   { select = false; }        // problems with amplitudes
  if (!simul){ 
    if (table)                                          { select = false; }        // moving table
    if ((qualX>2.0) || (qualX<0.0))                     { select = false; }        // quality cut 
    if ((qualY>2.0) || (qualY<0.0))                     { select = false; }
  }
  return select;
}


// to skip xtals without 3x3 around
bool atBorder(int xtalNum){

  bool amIatBd = false;
  if ( ((xtalNum-1)/20 == 0)  || ((xtalNum-1)/20 == 1)  ){amIatBd = true;}
  if ( ((xtalNum-1)/20 == 83) || ((xtalNum-1)/20 == 84) ){amIatBd = true;}
  if ( ((xtalNum-1)%20 == 0)  || ((xtalNum-1)%20 == 1)  ){amIatBd = true;}
  if ( ((xtalNum-1)%20 == 18) || ((xtalNum-1)%20 == 19) ){amIatBd = true;}
  return amIatBd;
}
