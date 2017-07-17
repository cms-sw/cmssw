#ifndef CaloAnalysis_CaloParticleFwd_h
#define CaloAnalysis_CaloParticleFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class CaloParticle;
typedef std::vector<CaloParticle> CaloParticleCollection;
typedef edm::Ref<CaloParticleCollection> CaloParticleRef;
typedef edm::RefVector<CaloParticleCollection> CaloParticleRefVector;
typedef edm::RefProd<CaloParticleCollection> CaloParticleRefProd;
typedef edm::RefVector<CaloParticleCollection> CaloParticleContainer;

std::ostream& operator<< (std::ostream& s, CaloParticle const & tp);

#endif
