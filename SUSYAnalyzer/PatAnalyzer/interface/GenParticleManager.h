#ifndef GEN_PARTICLE_MANAGER_H
#define GEN_PARTICLE_MANAGER_H

#include "Tools.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "TMath.h"
#include "PdgIdConverter.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
using namespace reco;

class GenParticleManager{
public:
    GenParticleManager(){};
    ~GenParticleManager(){};
    
    void SetCollection(edm::Handle<GenParticleCollection>& Col){ Collection = Col ; Reset();}
    void Classify();
    void Reset();
    
    void printInheritance(const GenParticle*);
    //getters
    std::vector<const GenParticle*> filterByStatus(std::vector<const GenParticle*>&, int status);
    std::vector<const GenParticle*>& getPromptMuons(){return vPromptMuons;}
    std::vector<const GenParticle*>& getNonPromptMuons(){return vNonPromptMuons;}
    std::vector<const GenParticle*>& getPromptElectrons(){return vPromptElectrons;}
    std::vector<const GenParticle*>& getNonPromptElectrons(){return vNonPromptElectrons;}
    std::vector<const GenParticle*>& getPromptTaus(){return vPromptTaus;}
    std::vector<const GenParticle*>& getNonPromptTaus(){return vNonPromptTaus;}
    
    std::vector<const GenParticle*>& getInvisible(){return vInvisible;}
    std::vector<const GenParticle*>& getZBosons(){return vZBosons;}
    std::vector<const GenParticle*>& getWBosons(){return vWBosons;}
    std::vector<const GenParticle*>& getHiggsBosons(){return vHiggsBosons;}
    std::vector<const GenParticle*>& getOffShellPhotons(){return vOffShellPhotons;}
    std::vector<const GenParticle*>& getCharginios(){return vCharginos;}
    std::vector<const GenParticle*>& getNeutralinos(){return vNeutralinos;}
    
    std::vector<const GenParticle*> getAllMothers(const GenParticle* p);
    
    int origin(const GenParticle* p );
    bool fromTop(const GenParticle* p );
    bool fromID(const GenParticle* p, const int pdgID );
    //template <class T> int origin(const T* p );
    int origin(const pat::Muon* p );
    int origin(const pat::Electron* p );
    int origin(const reco::PFTau* p );
    
    int originReduced(const int origin);

    const GenParticle* matchedMC(const pat::Muon *pReco);
    const GenParticle* matchedMC(const pat::Electron *pReco);
    
    bool comesFromBoson(std::vector<const GenParticle*>& );
    bool comesFromPhoton(std::vector<const GenParticle*>& );
    bool comesFromBBaryon(std::vector<const GenParticle*>& );
    bool comesFromCBaryon(std::vector<const GenParticle*>& );
    bool comesFromBMeson(std::vector<const GenParticle*>& );
    bool comesFromDMeson(std::vector<const GenParticle*>& );
    bool comesFromTau(std::vector<const GenParticle*>& );
    bool comesFromPi0(std::vector<const GenParticle*>& );
    bool comesFromUDS(std::vector<const GenParticle*>& );
    
    bool SameMother(const  GenParticle* p,  const GenParticle* part);
    
    //const GenParticle* LSP;
    bool isPrompt(const GenParticle* g);
    
    const GenParticle* getMother(const GenParticle* p);
    const GenParticle* getMother(const GenParticle* p, int i);
    const GenParticle* getMotherParton(const GenParticle* p);
    
protected:
    edm::Handle<GenParticleCollection> Collection;
    //leptons
    std::vector<const GenParticle*> vPromptMuons, vPromptElectrons, vPromptTaus;
    std::vector<const GenParticle*> vNonPromptMuons, vNonPromptElectrons, vNonPromptTaus;
    
    //bosons
    std::vector<const GenParticle*> vWBosons, vZBosons, vHiggsBosons, vOffShellPhotons;
    
    //EWK-inos
    std::vector<const GenParticle*> vCharginos, vNeutralinos;
    
    //stable invisible
    std::vector<const GenParticle*> vInvisible;
};


typedef GenParticleManager SIM;



#endif 
