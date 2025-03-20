#ifndef SimG4Core_Generator3_H
#define SimG4Core_Generator3_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Generators/interface/HepMCParticle.h"
#include "SimG4Core/Notification/interface/GenParticleInfo.h"

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"

#include <vector>

class G4Event;
class G4PrimaryParticle;
class LumiMonitorFilter;

class Generator3 {
public:
  Generator3(const edm::ParameterSet &p);
  virtual ~Generator3();

  void setGenEvent(const HepMC3::GenEvent *inpevt) {
    evt_ = (HepMC3::GenEvent *)inpevt;
    return;
  }
  void HepMC2G4(const HepMC3::GenEvent *g, G4Event *e);
  void nonCentralEvent2G4(const HepMC3::GenEvent *g, G4Event *e);
  virtual const HepMC3::GenEvent *genEvent() const { return evt_; }
  virtual const math::XYZTLorentzVector *genVertex() const { return vtx_; }
  virtual const double eventWeight() const { return weight_; }

private:
  bool particlePassesPrimaryCuts(const G4ThreeVector &p) const;
  bool isExotic(int pdgcode) const;
  bool isExoticNonDetectable(int pdgcode) const;
  bool IsInTheFilterList(int pdgcode) const;
  void particleAssignDaughters(G4PrimaryParticle *p, HepMC3::GenParticle *hp, double length);
  void setGenId(G4PrimaryParticle *p, int id) const { p->SetUserInformation(new GenParticleInfo(id)); }

private:
  bool fPCuts;
  bool fPtransCut;
  bool fEtaCuts;
  bool fPhiCuts;
  bool fFiductialCuts;
  bool fSlepton;
  double theMinPhiCut;
  double theMaxPhiCut;
  double theMinEtaCut;
  double theMaxEtaCut;
  double theMinPCut;
  double theMinPtCut2;
  double theMaxPCut;
  double theDecRCut2;
  double theEtaCutForHector;
  double theDecLenCut;
  double maxZCentralCMS;
  int verbose;
  LumiMonitorFilter *fLumiFilter;
  HepMC3::GenEvent *evt_;
  math::XYZTLorentzVector *vtx_;
  double weight_;
  double Z_lmin, Z_lmax, Z_hector;
  std::vector<int> pdgFilter;
  bool pdgFilterSel;
  bool fPDGFilter;
};

#endif
