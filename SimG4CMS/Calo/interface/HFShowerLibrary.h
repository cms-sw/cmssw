#ifndef SimG4CMS_HFShowerLibrary_h
#define SimG4CMS_HFShowerLibrary_h 1
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"

#include "G4ThreeVector.hh"

//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <string>
#include <memory>

class G4Step;
class G4ParticleTable;

class HFShowerLibrary {
public:
  //Constructor and Destructor
  HFShowerLibrary(const std::string &name,
                  const HcalDDDSimConstants *hcons,
                  const HcalSimulationParameters *hps,
                  edm::ParameterSet const &p);
  ~HFShowerLibrary();

public:
  struct Hit {
    Hit() {}
    G4ThreeVector position;
    int depth;
    double time;
  };

  std::vector<Hit> getHits(const G4Step *aStep, bool &ok, double weight, bool onlyLong = false);
  std::vector<Hit> fillHits(const G4ThreeVector &p,
                            const G4ThreeVector &v,
                            int parCode,
                            double parEnergy,
                            bool &ok,
                            double weight,
                            double time,
                            bool onlyLong = false);

  struct Params {
    double probMax_;
    double backProb_;
    double dphi_;
    bool equalizeTimeShift_;
    bool verbose_;
    bool applyFidCut_;
  };
  struct FileParams {
    std::string fileName_;
    std::string emBranchName_;
    std::string hadBranchName_;
    std::string branchEvInfo_;
    int fileVersion_;
  };
  HFShowerLibrary(Params const &, FileParams const &, HFFibre::Params);

private:
  HFShowerLibrary(const HcalDDDSimConstants *hcons,
                  const HcalSimulationParameters *hps,
                  edm::ParameterSet const &hfShower,
                  edm::ParameterSet const &hfShowerLibrary);
  bool rInside(double r) const;
  HFShowerPhotonCollection getRecord(int, int) const;
  void loadEventInfo(TBranch *);
  HFShowerPhotonCollection interpolate(int, double);
  HFShowerPhotonCollection extrapolate(int, double);
  void storePhoton(HFShowerPhoton const &iPhoton, HFShowerPhotonCollection &iPhotons) const;

  HFFibre fibre_;
  std::unique_ptr<TFile> hf_;
  TBranch *emBranch_, *hadBranch_;

  bool verbose_, applyFidCut_, newForm_, v3version_;
  int nMomBin_, totEvents_, evtPerBin_;
  float libVers_, listVersion_;
  std::vector<double> pmom_;

  int fileVersion_;
  bool equalizeTimeShift_;
  double probMax_, backProb_;
  double dphi_, rMin_, rMax_;
  std::vector<double> gpar_;
};
#endif
