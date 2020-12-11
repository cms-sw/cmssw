#ifndef CaloAnalysis_PFParticleFwd_h
#define CaloAnalysis_PFParticleFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

class PFParticle;
typedef std::vector<PFParticle> PFParticleCollection;
typedef edm::Ref<PFParticleCollection> PFParticleRef;
typedef edm::RefVector<PFParticleCollection> PFParticleRefVector;
typedef edm::RefProd<PFParticleCollection> PFParticleRefProd;
typedef edm::RefVector<PFParticleCollection> PFParticleContainer;

#endif

