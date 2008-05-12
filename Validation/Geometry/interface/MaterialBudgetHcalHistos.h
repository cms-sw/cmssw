#ifndef Validation_Geometry_MaterialBudgetHcalHistos_h
#define Validation_Geometry_MaterialBudgetHcalHistos_h 1

#include "Validation/Geometry/interface/TestHistoMgr.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <string>
#include <vector>

class MaterialBudgetHcalHistos {

public:
  
  MaterialBudgetHcalHistos(const edm::ParameterSet &p, TestHistoMgr* mgr);
  virtual ~MaterialBudgetHcalHistos() { hend(); }
  
  void fillBeginJob(const DDCompactView &);
  void fillStartTrack(const G4Track*);
  void fillPerStep(const G4Step *);
  void fillEndTrack();
  
private:
  
  void book(); 
  void fillHisto(int ii);
  void hend();
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<double>      getDDDArray(const std::string & str,
				       const DDsvalues_type & sv);
  bool isSensitive(std::string);
  bool isItHF(const G4VTouchable*);
  bool isItEC(std::string);
  
private:

  static const int         maxSet = 25;
  TestHistoMgr*            hmgr;
  std::string              theFileName;
  std::vector<std::string> sensitives, hfNames, sensitiveEC;
  std::vector<int>         hfLevels;
  int                      binEta, binPhi;
  double                   maxEta, etaLow, etaHigh;
  int                      id, layer, steps;
  double                   radLen, intLen, stepLen;
  double                   eta, phi;
};


#endif
