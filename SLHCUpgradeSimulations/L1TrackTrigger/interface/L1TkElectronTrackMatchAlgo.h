#ifndef L1TkElectronTrackMatchAlgo_HH
#define L1TkElectronTrackMatchAlgo_HH

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace L1TkElectronTrackMatchAlgo {
   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >    L1TkTrackCollectionType ;
  void doMatch(l1extra::L1EmParticleCollection::const_iterator egIter, const edm::Ptr< L1TkTrackType >& pTrk, double&  dph, double&  dr, double& deta);
  void doMatch(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk, double& dph, double&  dr, double& deta);

  double deltaR(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  double deltaPhi(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  double deltaEta(const GlobalPoint& epos, const edm::Ptr< L1TkTrackType >& pTrk);
  GlobalPoint calorimeterPosition(double phi, double eta, double e);

}  
#endif
