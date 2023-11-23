#include "TauAnalysis/MCEmbeddingTools/plugins/MuonDetCleaner.h"

#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

typedef MuonDetCleaner<CSCDetId, CSCRecHit2D> CSCRecHitColCleaner;
typedef MuonDetCleaner<CSCDetId, CSCSegment> CSCSegmentColCleaner;
typedef MuonDetCleaner<DTChamberId, DTRecSegment4D> DTRecSegment4DColCleaner;
typedef MuonDetCleaner<DTLayerId, DTRecHit1DPair> DTRecHitColCleaner;
typedef MuonDetCleaner<RPCDetId, RPCRecHit> RPCRecHitColCleaner;

//-------------------------------------------------------------------------------
// define 'getRawDetId' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T1, typename T2>
uint32_t MuonDetCleaner<T1, T2>::getRawDetId(const T2& recHit) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
}

template <>
uint32_t MuonDetCleaner<CSCDetId, CSCRecHit2D>::getRawDetId(const CSCRecHit2D& recHit) {
  return recHit.cscDetId().rawId();
}

template <>
uint32_t MuonDetCleaner<DTLayerId, DTRecHit1DPair>::getRawDetId(const DTRecHit1DPair& recHit) {
  return recHit.geographicalId().rawId();
}

template <>
uint32_t MuonDetCleaner<RPCDetId, RPCRecHit>::getRawDetId(const RPCRecHit& recHit) {
  return recHit.rpcId().rawId();
}

//-------------------------------------------------------------------------------
// find out what the kind of RecHit used by imput muons rechit
//-------------------------------------------------------------------------------

template <typename T1, typename T2>
bool MuonDetCleaner<T1, T2>::checkrecHit(const TrackingRecHit& recHit) {
  edm::LogError("TauEmbedding") << "!!!! Please add the checkrecHit for the individual class templates " assert(0);
}

template <>
bool MuonDetCleaner<CSCDetId, CSCRecHit2D>::checkrecHit(const TrackingRecHit& recHit) {
  const std::type_info& hit_type = typeid(recHit);
  if (hit_type == typeid(CSCSegment)) {
    return true;
  }  // This should be the default one (which are included in the global (outer) muon track)
  else if (hit_type == typeid(CSCRecHit2D)) {
    return true;
  }
  //else {std::cout<<"else "<<hit_type.name()<<std::endl;}
  return false;
}

template <>
uint32_t MuonDetCleaner<CSCDetId, CSCSegment>::getRawDetId(const CSCSegment& recHit)
{
  return recHit.cscDetId().rawId();
}

template <>
uint32_t MuonDetCleaner<DTChamberId, DTRecSegment4D>::getRawDetId(const DTRecSegment4D& recHit)
{
  return recHit.geographicalId().rawId();
}


template <>
bool MuonDetCleaner<DTLayerId, DTRecHit1DPair>::checkrecHit(const TrackingRecHit& recHit) {
  const std::type_info& hit_type = typeid(recHit);
  if (hit_type == typeid(DTRecSegment4D)) {
    return true;
  }  // This should be the default one (which are included in the global (outer) muon track)
  else if (hit_type == typeid(DTRecHit1D)) {
    return true;
  } else if (hit_type == typeid(DTSLRecCluster)) {
    return true;
  } else if (hit_type == typeid(DTSLRecSegment2D)) {
    return true;
  }
  // else {std::cout<<"else "<<hit_type.name()<<std::endl;}
  return false;
}

template <>
bool MuonDetCleaner<RPCDetId, RPCRecHit>::checkrecHit(const TrackingRecHit& recHit) {
  const std::type_info& hit_type = typeid(recHit);
  if (hit_type == typeid(RPCRecHit)) {
    return true;
  }  // This should be the default one (which are included in the global (outer) muon track)
  //else {std::cout<<"else "<<hit_type.name()<<std::endl;}
  return false;
}

template <>
bool MuonDetCleaner<CSCDetId, CSCSegment>::checkrecHit(const TrackingRecHit& recHit)
{	    
   const std::type_info &hit_type = typeid(recHit);
   if (hit_type == typeid(CSCSegment))  {return true;}  // This should be the default one (which are included in the global (outer) muon track)
   //else {std::cout<<"else "<<hit_type.name()<<std::endl;}    
   return false;
}

template <>
bool MuonDetCleaner<DTChamberId, DTRecSegment4D>::checkrecHit(const TrackingRecHit& recHit)
{	    
   const std::type_info &hit_type = typeid(recHit);
   if (hit_type == typeid(DTRecSegment4D))  {return true;}  // This should be the default one (which are included in the global (outer) muon track)
   else if (hit_type == typeid(DTRecHit1D)) {return true;}
   else if (hit_type == typeid(DTSLRecCluster)) {return true; }
   else if (hit_type == typeid(DTSLRecSegment2D)) {return true; }
  // else {std::cout<<"else "<<hit_type.name()<<std::endl;}	    
   return false;
}

DEFINE_FWK_MODULE(CSCRecHitColCleaner);
DEFINE_FWK_MODULE(DTRecHitColCleaner);
DEFINE_FWK_MODULE(RPCRecHitColCleaner);
DEFINE_FWK_MODULE(CSCSegmentColCleaner);
DEFINE_FWK_MODULE(DTRecSegment4DColCleaner);