#ifndef SimG4Core_Generator_H
#define SimG4Core_Generator_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Generators/interface/HepMCParticle.h"
#include "SimG4Core/Notification/interface/GenParticleInfo.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenParticle.h"
#include "CLHEP/HepMC/ParticleDataTableConfig.h"

#include "CLHEP/Vector/LorentzVector.h"

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
    virtual const HepLorentzVector* genVertex() const { return vtx_; }
    virtual const double eventWeight() const { return weight_; }
private:
    bool particlePassesPrimaryCuts(const G4PrimaryParticle * p) const;
    bool particlePassesPrimaryCuts( const HepLorentzVector& mom ) const ;
    void particleAssignDaughters(G4PrimaryParticle * p, HepMC::GenParticle * hp, double length);
    void setGenId(G4PrimaryParticle* p, int id) const 
      {p->SetUserInformation(new GenParticleInfo(id));}

private:
    bool   fPtCuts;
    bool   fEtaCuts;
    bool   fPhiCuts;
    double theMinPhiCut;
    double theMaxPhiCut;
    double theMinEtaCut;
    double theMaxEtaCut;
    double theMinPtCut;
    double theMaxPtCut;
    double theDecLenCut;
    int verbose;
    HepMC::GenEvent*  evt_;
    HepLorentzVector* vtx_;
    double weight_;    
};

#endif
