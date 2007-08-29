#ifndef SimDataFormats_JetMatching_JetFlavour_H
#define SimDataFormats_JetMatching_JetFlavour_H

#include <vector>
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"

namespace reco
{
/**
 * JetFlavour class is meant to be used when the genEvent is dropped.
 * It can store by value the matching information about flavour and parton kinematics
 * The flavour definition and the corresponding parton information should be configured
 * in the producer.
 * */
class JetFlavour
{
  public:
    JetFlavour();
     
  private:
    LorenzVector m_partonMomentum;
    Point m_partonVertex; //is it needed?
    int m_flavour;

};
}
#endif
