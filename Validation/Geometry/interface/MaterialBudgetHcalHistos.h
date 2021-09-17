#ifndef Validation_Geometry_MaterialBudgetHcalHistos_h
#define Validation_Geometry_MaterialBudgetHcalHistos_h 1

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
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
  void fillBeginJob(const cms::DDCompactView &);
  void fillStartTrack(const G4Track *);
  void fillPerStep(const G4Step *);
  void fillEndTrack();

private:
  void book();
  void fillHisto(int ii);
  void fillLayer();
  void hend();
  std::vector<std::string> getNames(DDFilteredView &fv);
  std::vector<std::string> getNames(cms::DDFilteredView &fv);
  std::vector<double> getDDDArray(const std::string &str, const DDsvalues_type &sv);
  bool isSensitive(const std::string &);
  bool isItHF(const G4VTouchable *);
  bool isItEC(const std::string &);

private:
  static const int maxSet_ = 25, maxSet2_ = 9;
  std::vector<std::string> sensitives_, hfNames_, sensitiveEC_;
  std::vector<int> hfLevels_;
  bool fillHistos_, printSum_, fromdd4hep_;
  int binEta_, binPhi_;
  double maxEta_, etaLow_, etaHigh_, etaLowMin_, etaLowMax_, etaMidMin_;
  double etaMidMax_, etaHighMin_, etaHighMax_, etaMinP_, etaMaxP_;
  std::vector<std::string> matList_;
  std::vector<double> stepLength_, radLength_, intLength_;
  TH1F *me400[maxSet_], *me800[maxSet_], *me1300[maxSet2_];
  TH2F *me1200[maxSet_], *me1400[maxSet2_];
  TProfile *me100[maxSet_], *me200[maxSet_], *me300[maxSet_];
  TProfile *me500[maxSet_], *me600[maxSet_], *me700[maxSet_];
  TProfile *me1500[maxSet2_];
  TProfile *me1600[maxSet_], *me1700[maxSet_], *me1800[maxSet_];
  TProfile *me1900[maxSet_], *me2000[maxSet_], *me2100[maxSet_];
  TProfile *me2200[maxSet_], *me2300[maxSet_], *me2400[maxSet_];
  TProfile2D *me900[maxSet_], *me1000[maxSet_], *me1100[maxSet_];
  int id_, layer_, steps_;
  double radLen_, intLen_, stepLen_;
  double eta_, phi_;
  int nlayHB_, nlayHE_, nlayHO_, nlayHF_;
};

#endif
