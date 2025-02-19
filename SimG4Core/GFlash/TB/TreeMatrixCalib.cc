#include "SimG4Core/GFlash/TB/TreeMatrixCalib.h"

TreeMatrixCalib::TreeMatrixCalib(const char * filename) 
{
  myFile = new TFile(filename,"RECREATE");
  myTree = new TTree("T1","my tree");

  // Amplitude / hodoscopes / tdc infos
  myTree->Branch("run",             &myRun,            "run/I");  
  myTree->Branch("event",           &myEvent,          "event/I");  
  myTree->Branch("xtalSM",          &myXtalSM,         "xtalSM/I");  
  myTree->Branch("maxEneXtal",      &myMaxEneXtal,     "maxEneXtal/I");  
  myTree->Branch("nominalXtalSM",   &myNominalXtalSM,  "nominalXtalSM/I");  
  myTree->Branch("nextXtalSM",      &myNextXtalSM,     "nextXtalSM/I");  
  myTree->Branch("xtalEta",         &myXtalEta,        "xtalEta/I");  
  myTree->Branch("xtalPhi",         &myXtalPhi,        "xtalPhi/I");  
  myTree->Branch("tbMoving",        &myTbMoving,       "tbMoving/I");
  myTree->Branch("hodoX",           &myHodoX,          "hodoX/D");  
  myTree->Branch("hodoY",           &myHodoY,          "hodoY/D");  
  myTree->Branch("caloX",           &myCaloX,          "caloX/D");  
  myTree->Branch("caloY",           &myCaloY,          "caloY/D");  
  myTree->Branch("hodoSlopeX",      &myHodoSlopeX,     "hodoSlopeX/D");  
  myTree->Branch("hodoSlopeY",      &myHodoSlopeY,     "hodoSlopeY/D");  
  myTree->Branch("hodoQualityX",    &myHodoQualityX,   "hodoQualityX/D");  
  myTree->Branch("hodoQualityY",    &myHodoQualityY,   "hodoQualityY/D");  
  myTree->Branch("tdcOffset",       &myTdcOffset,      "tdcOffset/D");  
  myTree->Branch("allMatrix",       &myAllMatrix,      "allMatrix/I");
  myTree->Branch("amplit",          &myAmplit,         "amplit[49]/D");    
  myTree->Branch("crystal",         &myCrystal,        "crystal[49]/I");  
}

TreeMatrixCalib::~TreeMatrixCalib() 
{
  myFile->cd();
  myTree->Write();
  myFile->Close();
  delete myFile;
}

void TreeMatrixCalib::store()
{
  myTree->Fill();
}

void TreeMatrixCalib::fillInfo( int run, int eve, int xnum, int maxX, int nomX, int nextX, int xeta, int xphi, int tbm, double xx, double yy, double ecalx, double ecaly, double sx, double sy, double qx, double qy, double tdcoff, int allm, double amp[], int cry[])
{
  myRun           = run;
  myEvent         = eve;
  myXtalSM        = xnum;
  myMaxEneXtal    = maxX;
  myNominalXtalSM = nomX;
  myNextXtalSM    = nextX;
  myXtalEta       = xeta;
  myXtalPhi       = xphi;
  myTbMoving      = tbm;
  myHodoX         = xx;
  myHodoY         = yy;
  myCaloX         = ecalx;
  myCaloY         = ecaly;
  myHodoSlopeX    = sx;
  myHodoSlopeY    = sy;
  myHodoQualityX  = qx;
  myHodoQualityY  = qy;
  myTdcOffset     = tdcoff;
  myAllMatrix     = allm;
  for (int ii=0; ii<49; ii++){
    myAmplit[ii]  = amp[ii]; 
    myCrystal[ii] = cry[ii]; 
  }
}

















