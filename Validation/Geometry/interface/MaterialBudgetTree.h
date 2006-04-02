#ifndef MaterialBudgetTree_h
#define MaterialBudgetTree_h 1

#include "TFile.h"
#include "TTree.h"
#include "G4ThreeVector.hh"

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"

//#include "HTL/Histograms.h" // Transient histograms.


class MaterialBudgetTree : public MaterialBudgetFormat
{
public:

  MaterialBudgetTree( MaterialBudgetData* data, const std::string& fileName );   
  virtual ~MaterialBudgetTree(){ hend(); }

  virtual void fillStartTrack();
  virtual void fillPerStep();
  virtual void fillEndTrack();
  
private:
  
  virtual void book();  // user booking
  virtual void hend();  // user ending
  
 private:
  TFile * theFile;
  TTree* theTree; 

  static const int MAXSTEPS = 5000;
  float t_MB;
  float t_Eta;
  float t_Phi;
  int t_Nsteps;
  float t_DeltaMB[MAXSTEPS];
  float t_X[MAXSTEPS];
  float t_Y[MAXSTEPS];
  float t_Z[MAXSTEPS];
  int t_VoluID[MAXSTEPS];
  int t_MateID[MAXSTEPS];

};


#endif
