#ifndef SimG4Core_Generator_H
#define SimG4Core_Generator_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Generators/interface/HepMCParticle.h"
#include "SimG4Core/Notification/interface/GenParticleInfo.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>
#include <map>
#include <string>
    
class G4Event;
class G4PrimaryParticle;

class Generator
{
public:
    Generator(const edm::ParameterSet & p);
    virtual ~Generator();
    // temp.(?) method
    void setGenEvent( const HepMC::GenEvent* inpevt ) { evt_ = (HepMC::GenEvent*)inpevt; return ; }
    void HepMC2G4(const HepMC::GenEvent * g,G4Event * e);
    void nonBeamEvent2G4(const HepMC::GenEvent * g,G4Event * e);
    virtual const HepMC::GenEvent*  genEvent() const { return evt_; }
    virtual const math::XYZTLorentzVector* genVertex() const { return vtx_; }
    virtual const double eventWeight() const { return weight_; }
private:
    bool particlePassesPrimaryCuts(const G4PrimaryParticle * p) const;
    bool particlePassesPrimaryCuts( const math::XYZTLorentzVector& mom, const double zimp ) const ;
    void particleAssignDaughters(G4PrimaryParticle * p, HepMC::GenParticle * hp, double length);
    void setGenId(G4PrimaryParticle* p, int id) const 
      {p->SetUserInformation(new GenParticleInfo(id));}

private:
  bool   fPCuts;
  bool   fEtaCuts;
  bool   fPhiCuts;
  double theMinPhiCut;
  double theMaxPhiCut;
  double theMinEtaCut;
  double theMaxEtaCut;
  double theMinPCut;
  double theMaxPCut;
  double theRDecLenCut;
  double theEtaCutForHector; 
  int verbose;
  HepMC::GenEvent*  evt_;
  math::XYZTLorentzVector* vtx_;
  double weight_;    
  double Z_lmin,Z_lmax,Z_hector;
  std::vector<int> pdgFilter;
  bool pdgFilterSel;
};

#endif
