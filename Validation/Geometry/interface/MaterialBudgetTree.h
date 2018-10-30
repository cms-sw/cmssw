#ifndef MaterialBudgetTree_h
#define MaterialBudgetTree_h 1

#include "TFile.h"
#include "TTree.h"
#include "G4ThreeVector.hh"

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"

class MaterialBudgetTree : public MaterialBudgetFormat
{
public:

  MaterialBudgetTree( std::shared_ptr<MaterialBudgetData> data, const std::string& fileName );   
  ~MaterialBudgetTree() override { }

  void fillStartTrack() override;
  void fillPerStep() override;
  void fillEndTrack() override;
  void endOfRun() override;
  
private:
  
  void book();  // user booking
  std::unique_ptr<TFile> theFile;
  std::unique_ptr<TTree> theTree;

  static const int MAXSTEPS = 10000;

  float t_MB;
  float t_IL;

  int   t_ParticleID;
  float t_ParticlePt;
  float t_ParticleEta;
  float t_ParticlePhi;
  float t_ParticleEnergy;
  float t_ParticleMass;

  int t_Nsteps;

  float t_DeltaMB[MAXSTEPS];
  float t_DeltaMB_SUP[MAXSTEPS];
  float t_DeltaMB_SEN[MAXSTEPS];
  float t_DeltaMB_CAB[MAXSTEPS];
  float t_DeltaMB_COL[MAXSTEPS];
  float t_DeltaMB_ELE[MAXSTEPS];
  float t_DeltaMB_OTH[MAXSTEPS];
  float t_DeltaMB_AIR[MAXSTEPS];

  float t_DeltaIL[MAXSTEPS];
  float t_DeltaIL_SUP[MAXSTEPS];
  float t_DeltaIL_SEN[MAXSTEPS];
  float t_DeltaIL_CAB[MAXSTEPS];
  float t_DeltaIL_COL[MAXSTEPS];
  float t_DeltaIL_ELE[MAXSTEPS];
  float t_DeltaIL_OTH[MAXSTEPS];
  float t_DeltaIL_AIR[MAXSTEPS];

  double t_InitialX[MAXSTEPS];
  double t_InitialY[MAXSTEPS];
  double t_InitialZ[MAXSTEPS];
  double t_FinalX[MAXSTEPS];
  double t_FinalY[MAXSTEPS];
  double t_FinalZ[MAXSTEPS];

  int    t_VolumeID[MAXSTEPS];
  const char*  t_VolumeName[MAXSTEPS];
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
  const char* t_MaterialName[MAXSTEPS];  
  float t_MaterialX0[MAXSTEPS];  
  float t_MaterialLambda0[MAXSTEPS];  
  float t_MaterialDensity[MAXSTEPS]; // g/cm3  
  int   t_ParticleStepID[MAXSTEPS];  
  float t_ParticleStepInitialPt[MAXSTEPS];  
  float t_ParticleStepInitialEta[MAXSTEPS];  
  float t_ParticleStepInitialPhi[MAXSTEPS];  
  float t_ParticleStepInitialEnergy[MAXSTEPS];  
  float t_ParticleStepInitialPx[MAXSTEPS];  
  float t_ParticleStepInitialPy[MAXSTEPS];  
  float t_ParticleStepInitialPz[MAXSTEPS];  
  float t_ParticleStepInitialBeta[MAXSTEPS];  
  float t_ParticleStepInitialGamma[MAXSTEPS];  
  float t_ParticleStepInitialMass[MAXSTEPS];  
  float t_ParticleStepFinalPt[MAXSTEPS];  
  float t_ParticleStepFinalEta[MAXSTEPS];  
  float t_ParticleStepFinalPhi[MAXSTEPS];  
  float t_ParticleStepFinalEnergy[MAXSTEPS];  
  float t_ParticleStepFinalPx[MAXSTEPS];  
  float t_ParticleStepFinalPy[MAXSTEPS];  
  float t_ParticleStepFinalPz[MAXSTEPS];  
  float t_ParticleStepFinalBeta[MAXSTEPS];  
  float t_ParticleStepFinalGamma[MAXSTEPS];  
  float t_ParticleStepFinalMass[MAXSTEPS];  
  int   t_ParticleStepPreInteraction[MAXSTEPS];  
  int   t_ParticleStepPostInteraction[MAXSTEPS];  

};

#endif
