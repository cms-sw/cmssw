#ifndef SimDataFormats_JetMatching_JetFlavourInfo_H
#define SimDataFormats_JetMatching_JetFlavourInfo_H

#include <vector>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

namespace reco
{
/**\class JetFlavourInfo JetFlavourInfo.h SimDataFormats/JetMatching/interface/JetFlavourInfo.h
 * \brief Class storing the jet flavour information
 *
 * JetFlavourInfo class stores the jet flavour information based on hadrons
 * and partons clustered inside the jet. It also provides vectors of
 * EDM references to clustered hadrons and partons. The hadron- and parton-based
 * flavours are defined in the JetFlavourClustering producer.
 */

class JetFlavourInfo
{
  public:
    JetFlavourInfo(void) { }
    JetFlavourInfo(
                   const GenParticleRefVector& bHadrons,
                   const GenParticleRefVector& cHadrons,
                   const GenParticleRefVector& partons,
                   const GenParticleRefVector& leptons,
                   const int hadronFlavour,
                   const int partonFlavour
                  ) :
      m_bHadrons(bHadrons),
      m_cHadrons(cHadrons),
      m_partons(partons),
      m_leptons(leptons),
      m_hadronFlavour(hadronFlavour),
      m_partonFlavour(partonFlavour) { }

    /// Return a vector of GenParticleRef's to b hadrons clustered inside the jet
    const GenParticleRefVector & getbHadrons() const { return m_bHadrons; }
    /// Return a vector of GenParticleRef's to c hadrons clustered inside the jet
    const GenParticleRefVector & getcHadrons() const { return m_cHadrons; }
    /// Return a vector of GenParticleRef's to partons clustered inside the jet
    const GenParticleRefVector & getPartons() const { return m_partons; }
    /// Return a vector of GenParticleRef's to leptons clustered inside the jet
    const GenParticleRefVector & getLeptons() const { return m_leptons; }
    /// Return the hadron-based flavour
    const int getHadronFlavour() const { return m_hadronFlavour; }
    /// Return the parton-based flavour
    const int getPartonFlavour() const { return m_partonFlavour; }

  private:
    GenParticleRefVector m_bHadrons;
    GenParticleRefVector m_cHadrons;
    GenParticleRefVector m_partons;
    GenParticleRefVector m_leptons;
    int m_hadronFlavour;
    int m_partonFlavour;
};

}
#endif
