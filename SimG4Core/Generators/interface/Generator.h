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
    typedef std::map<HepMC::GenParticle *,HepMCParticle *> ParticleMapType;
    typedef ParticleMapType::const_iterator PMT;     
    Generator(const edm::ParameterSet & p);
    virtual ~Generator();
    // temp.method
    void setGenEvent( const HepMC::GenEvent* inpevt ) { evt_ = (HepMC::GenEvent*)inpevt; return ; }
    const HepMC::GenEvent * generateEvent() ;
    void HepMC2G4(const HepMC::GenEvent * g,G4Event * e);
    // temp.method
    std::string genInputType() const { return inputFileName ; }
    virtual const HepMC::GenEvent * genEvent() const { return evt_; }
    virtual const HepLorentzVector genVertex() const { return vtx_; }
    virtual const double eventWeight() const { return weight_; }
    virtual void  runNumber(int r) { runNumber_ = r; }
    virtual const int runNumber() const { return runNumber_; }
private:
    bool particlePassesPrimaryCuts(const G4PrimaryParticle * p) const;
    void setGenId(G4PrimaryParticle* p, int id) const 
    { p->SetUserInformation(new GenParticleInfo(id)); }
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
    std::string inputFileName;
    int verbose;
    HepMC::GenEvent * evt_;
    HepLorentzVector vtx_;
    double weight_;    
    int runNumber_;
    ParticleMapType pmap; 
};

#endif
