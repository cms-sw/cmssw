#ifndef CaloAnalysis_MtdCaloParticleFwd_h
#define CaloAnalysis_MtdCaloParticleFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

class MtdCaloParticle;
std::ostream &operator<<(std::ostream &s, MtdCaloParticle const &tp);

typedef std::vector<MtdCaloParticle> MtdCaloParticleCollection;
typedef edm::Ref<MtdCaloParticleCollection> MtdCaloParticleRef;
typedef edm::RefVector<MtdCaloParticleCollection> MtdCaloParticleRefVector;
typedef edm::RefProd<MtdCaloParticleCollection> MtdCaloParticleRefProd;
typedef edm::RefVector<MtdCaloParticleCollection> MtdCaloParticleContainer;

#endif
