#include "TauAnalysis/MCEmbeddingTools/plugins/MuonDetRecHitMixer.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

typedef MuonDetRecHitMixer<CSCDetId, CSCRecHit2D> CSCRecHitMixer;
typedef MuonDetRecHitMixer<DTLayerId, DTRecHit1DPair> DTRecHitMixer;
typedef MuonDetRecHitMixer<RPCDetId, RPCRecHit> RPCRecHitMixer;

//-------------------------------------------------------------------------------
// define 'getDetIds' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T1, typename T2>
uint32_t MuonDetRecHitMixer<T1,T2>::getRawDetId(const T2& recHit)
{
  assert(0); // CV: make sure general function never gets called;
             //     always use template specializations
}


template <>
uint32_t MuonDetRecHitMixer<CSCDetId, CSCRecHit2D>::getRawDetId(const CSCRecHit2D& recHit)
{
  return recHit.cscDetId().rawId();
}

template <>
uint32_t MuonDetRecHitMixer<DTLayerId, DTRecHit1DPair>::getRawDetId(const DTRecHit1DPair& recHit)
{
  return recHit.geographicalId().rawId();
}

template <>
uint32_t MuonDetRecHitMixer<RPCDetId, RPCRecHit>::getRawDetId(const RPCRecHit& recHit)
{
  return recHit.rpcId().rawId();
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CSCRecHitMixer);
DEFINE_FWK_MODULE(DTRecHitMixer);
DEFINE_FWK_MODULE(RPCRecHitMixer);



