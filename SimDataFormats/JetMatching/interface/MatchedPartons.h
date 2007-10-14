#ifndef MatchedParton_H
#define MatchedParton_H

#include <vector>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco
{

class MatchedPartons
{
  public:
    
    MatchedPartons() { }
    MatchedPartons( 
                    CandidateRef hv,
                    CandidateRef n2,
                    CandidateRef n3,
                    CandidateRef pd,
                    CandidateRef ad 
                  ) : 
       m_heaviest(hv),
       m_nearest2(n2),
       m_nearest3(n3),
       m_PhysDef(pd),
       m_AlgoDef(ad) { }

    //Return the ParticleRef for the heaviest flavour in the signal cone
    const CandidateRef heaviest() const {return m_heaviest;}

    //Return the ParticleRef for the nearest parton (status=2)
    const CandidateRef & nearest_status2() const {return m_nearest2;}

    //Return the ParticleRef for the nearest parton (status=3)
    const CandidateRef & nearest_status3() const {return m_nearest3;}
    
    //Return the ParticleRef for the Physics Definition parton
    const CandidateRef & physicsDefinitionParton() const {return m_PhysDef;}

    //Return the ParticleRef for the Algorithmic Definition parton
    const CandidateRef & algoDefinitionParton() const {return m_AlgoDef;}

  private:

    CandidateRef m_heaviest;
    CandidateRef m_nearest2;
    CandidateRef m_nearest3;
    CandidateRef m_PhysDef;
    CandidateRef m_AlgoDef;

};

}
#endif
