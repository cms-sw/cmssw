#ifndef Validation_Geometry_MaterialBudgetHcalHistos_h
#define Validation_Geometry_MaterialBudgetHcalHistos_h 1

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <string>
#include <vector>

class MaterialBudgetHcalHistos {

public:
  
  MaterialBudgetHcalHistos(const edm::ParameterSet &p);
  virtual ~MaterialBudgetHcalHistos() { hend(); }
  
  void fillBeginJob(const DDCompactView &);
  void fillStartTrack(const G4Track*);
  void fillPerStep(const G4Step *);
  void fillEndTrack();
  
private:
  
  void book(); 
  void fillHisto(int ii);
  void fillLayer();
  void hend();
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<double>      getDDDArray(const std::string & str,
				       const DDsvalues_type & sv);
  bool isSensitive(std::string);
  bool isItHF(const G4VTouchable*);
  bool isItEC(std::string);
  
private:

  static const int         maxSet = 25, maxSet2 = 9;
  std::vector<std::string> sensitives, hfNames, sensitiveEC;
  std::vector<int>         hfLevels;
  bool                     fillHistos, printSum;
  int                      binEta, binPhi;
  double                   maxEta, etaLow, etaHigh;
  std::vector<std::string> matList;
  std::vector<double>      stepLength, radLength, intLength;
  TH1F                     *me400[maxSet], *me800[maxSet], *me1300[maxSet2];
  TH2F                     *me1200[maxSet],*me1400[maxSet2];
  TProfile                 *me100[maxSet], *me200[maxSet], *me300[maxSet];
  TProfile                 *me500[maxSet], *me600[maxSet], *me700[maxSet];
  TProfile                 *me1500[maxSet2];
  TProfile2D               *me900[maxSet], *me1000[maxSet],*me1100[maxSet];
  int                      id, layer, steps;
  double                   radLen, intLen, stepLen;
  double                   eta, phi;
  int                      nlayHB, nlayHE, nlayHO, nlayHF;
};


#endif
