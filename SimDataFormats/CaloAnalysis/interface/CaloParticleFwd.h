#ifndef CaloAnalysis_CaloParticleFwd_h
#define CaloAnalysis_CaloParticleFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace io_v1 {
  class CaloParticle;
  std::ostream &operator<<(std::ostream &s, CaloParticle const &tp);
}  // namespace io_v1
using CaloParticle = io_v1::CaloParticle;

typedef std::vector<CaloParticle> CaloParticleCollection;
typedef edm::Ref<CaloParticleCollection> CaloParticleRef;
typedef edm::RefVector<CaloParticleCollection> CaloParticleRefVector;
typedef edm::RefProd<CaloParticleCollection> CaloParticleRefProd;
typedef edm::RefVector<CaloParticleCollection> CaloParticleContainer;

#endif
