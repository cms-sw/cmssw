#ifndef Validation_Geometry_MaterialBudgetCastorHistos_h
#define Validation_Geometry_MaterialBudgetCastorHistos_h 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <string>
#include <vector>

class MaterialBudgetCastorHistos {

public:
  
  MaterialBudgetCastorHistos(const edm::ParameterSet &p);
  virtual ~MaterialBudgetCastorHistos();
  
  void fillStartTrack(const G4Track*);
  void fillPerStep(const G4Step *);
  void fillEndTrack();
  
private:
  
  void book(); 
  void fillHisto(int id, int ix);
  
private:

  static const int         maxSet = 20;
  bool                     fillHistos, printSum;
  int                      binEta, binPhi;
  double                   etaLow, etaHigh;
  std::vector<std::string> matList;
  std::vector<double>      stepLength, radLength, intLength;
  TH1F                     *me400[maxSet], *me800[maxSet];
  TH2F                     *me1200[maxSet];
  TProfile                 *me100[maxSet], *me200[maxSet], *me300[maxSet];
  TProfile                 *me500[maxSet], *me600[maxSet], *me700[maxSet];
  TProfile2D               *me900[maxSet], *me1000[maxSet],*me1100[maxSet];
  int                      id1, id2, steps;
  double                   radLen, intLen, stepLen;
  double                   eta, phi;
};


#endif
