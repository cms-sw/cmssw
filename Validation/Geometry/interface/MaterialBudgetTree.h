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
  //  float t_Eta;
  //  float t_Phi;
  // rr
  int   t_ParticleID;
  float t_ParticlePt;
  float t_ParticleEta;
  float t_ParticlePhi;
  float t_ParticleEnergy;
  // rr
  int t_Nsteps;
  float t_DeltaMB[MAXSTEPS];
  float t_DeltaMB_SUP[MAXSTEPS];
  float t_DeltaMB_SEN[MAXSTEPS];
  float t_DeltaMB_CAB[MAXSTEPS];
  float t_DeltaMB_COL[MAXSTEPS];
  float t_DeltaMB_ELE[MAXSTEPS];
  float t_DeltaMB_OTH[MAXSTEPS];
  float t_DeltaMB_AIR[MAXSTEPS];
  double t_InitialX[MAXSTEPS];
  double t_InitialY[MAXSTEPS];
  double t_InitialZ[MAXSTEPS];
  double t_FinalX[MAXSTEPS];
  double t_FinalY[MAXSTEPS];
  double t_FinalZ[MAXSTEPS];
  // rr
  int    t_VolumeID[MAXSTEPS];
  char*  t_VolumeName[MAXSTEPS];
  int    t_VolumeCopy[MAXSTEPS];
  float  t_VolumeX[MAXSTEPS];
  float  t_VolumeY[MAXSTEPS];
  float  t_VolumeZ[MAXSTEPS];
  float  t_VolumeXaxis1[MAXSTEPS];
  float  t_VolumeXaxis2[MAXSTEPS];
  float  t_VolumeXaxis3[MAXSTEPS];
  float  t_VolumeYaxis1[MAXSTEPS];
  float  t_VolumeYaxis2[MAXSTEPS];
  float  t_VolumeYaxis3[MAXSTEPS];
  float  t_VolumeZaxis1[MAXSTEPS];
  float  t_VolumeZaxis2[MAXSTEPS];
  float  t_VolumeZaxis3[MAXSTEPS];
  int   t_MaterialID[MAXSTEPS];
  char* t_MaterialName[MAXSTEPS];  
  float t_MaterialX0[MAXSTEPS];  
  int   t_ParticleStepID[MAXSTEPS];  
  float t_ParticleStepInitialPt[MAXSTEPS];  
  float t_ParticleStepInitialEta[MAXSTEPS];  
  float t_ParticleStepInitialPhi[MAXSTEPS];  
  float t_ParticleStepInitialEnergy[MAXSTEPS];  
  float t_ParticleStepFinalPt[MAXSTEPS];  
  float t_ParticleStepFinalEta[MAXSTEPS];  
  float t_ParticleStepFinalPhi[MAXSTEPS];  
  float t_ParticleStepFinalEnergy[MAXSTEPS];  
  int   t_ParticleStepInteraction[MAXSTEPS];  
  // rr
};


#endif
