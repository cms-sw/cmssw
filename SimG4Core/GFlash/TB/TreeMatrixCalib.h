#ifndef TreeMatrixCalib_h
#define TreeMatrixCalib_h

// includes
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"

class TFile;
class TTree;

class G3EventProxy;

class TreeMatrixCalib {
public:
   TreeMatrixCalib(const char * filename = "tb.root"); 
  ~TreeMatrixCalib(); 

  void fillInfo( int run, int eve, int xnum, int maxX, int nomX, int nextX, int xeta, int xphi, int tbm, double xx, double yy, double ecalx, double ecaly, double sx, double sy, double qx, double qy, double tdcoff, int allm, double amp[], int cry[]);

  void store();
  

 private:
  
  TFile* myFile;
  TTree* myTree;

  // general info
  int myEvent, myRun;
  int myXtalSM,   myXtalEta,  myXtalPhi;
  int myNominalXtalSM, myNextXtalSM;
  int myTbMoving;
  int myMaxEneXtal;

  // amplitude 
  double myAmplit[49];

  // crystals
  int myCrystal[49];

  // hodoscope infos  
  double myHodoX,        myHodoY;
  double myHodoSlopeX,   myHodoSlopeY;
  double myHodoQualityX, myHodoQualityY;

  // ecal position
  double myCaloX, myCaloY;  

  // tdc info
  double myTdcOffset;

  // all matrix
  int myAllMatrix;
  
};

#endif
