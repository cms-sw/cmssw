#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

namespace SimDataFormats {
  namespace CaloAnalysis {
    SimCluster sc;
    SimClusterCollection vsc;
    edm::Wrapper<SimClusterCollection> wvsc;

    SimClusterRef scr;
    SimClusterRefVector scrv;
    SimClusterRefProd scrp;
    SimClusterContainer scc;

    CaloParticle cp;
    CaloParticleCollection vcp;
    edm::Wrapper<CaloParticleCollection> wvcp;

    CaloParticleRef cpr;
    CaloParticleRefVector cprv;
    CaloParticleRefProd cprp;
    CaloParticleContainer cpc;
  }  // namespace CaloAnalysis
}  // namespace SimDataFormats
