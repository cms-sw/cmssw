#ifndef SimDataFormats_JetMatching_JetFlavour_H
#define SimDataFormats_JetMatching_JetFlavour_H

#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace reco
{
/**
 * JetFlavour class is meant to be used when the genEvent is dropped.
 * It can store by value the matching information about flavour and parton kinematics
 * The flavour definition and the corresponding parton information should be configured
 * in the producer.
 * The typedefs are taken from reco::Particle
 * */
class JetFlavour
{
  public:
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;

    JetFlavour(void) { }
     
  private:
    LorentzVector m_partonMomentum;
    Point         m_partonVertex;       // is it needed?
    int           m_flavour;
};

}
#endif
