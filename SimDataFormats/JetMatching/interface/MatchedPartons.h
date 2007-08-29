#ifndef MatchedParton_H
#define MatchedParton_H

#include <vector>
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
namespace reco
{
class MatchedPartons
{
  public:
    MatchedPartons();
    MatchedPartons(reco::GenParticleCandidateRefVector partons,
    std::vector<float> deltaR) : m_partons(partons), m_deltaR(deltaR) 
    {}

   /** 
      Return the GenParticleRef for the heaviest flavour in the signal cone
    */ 
    const reco::GenParticleCandidateRef & heaviest(float signalCone = 0.3) const;

   /**
      Return the GenParticleRef for the heaviest flavour in the signal cone
    */ 
    const reco::GenParticleCandidateRef & physicsDefinitionParton(float signalCone = 0.3) const;

   /**
      Return the GenParticleRef for the nearest parton
    */ 
    const reco::GenParticleCandidateRef & nearest() const;
    
   /**
     Return the list of partons associated, ordered by distance
    */
    reco::GenParticleCandidateRefVector partons() {return m_partons;}
   /**
     Return the list of partons deltaR to the original jet
    */
    std::vector<float> deltaR() {return m_deltaR;}
     
  private:
    reco::GenParticleCandidateRefvector m_partons;
    std::vector<float> m_deltaR;


};
}
#endif
