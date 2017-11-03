#ifndef __TMTrackTrigger_VertexFinder_L1fittedTrack_h__
#define __TMTrackTrigger_VertexFinder_L1fittedTrack_h__


#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include <vector>


class TrackerGeometry;
class TrackerTopology;


namespace vertexFinder {

class Settings;
class Stub;
class TP;

typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_>           TTStubAssMap;
typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_>        TTClusterAssMap;

//! Simple wrapper class for TTTrack, to avoid changing other areas of packages immediately
class L1fittedTrack {
public:
  L1fittedTrack(const TTTrack< Ref_Phase2TrackerDigi_ >&, const Settings& , const TrackerGeometry* , const TrackerTopology*, const std::map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, edm::Handle<TTStubAssMap> mcTruthTTStubHandle, edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle);
  ~L1fittedTrack();

  float eta() const;
  float phi0() const;
  float pt() const;
  float z0() const;

  float chi2dof() const;
  unsigned int getNumStubs()  const  {return numStubs;}

  // Get best matching tracking particle (=nullptr if none).
  const TP* getMatchedTP() const;

private:
  TTTrack< Ref_Phase2TrackerDigi_ > track_;

  //--- Information about its association (if any) to a truth Tracking Particle.
  const TP*             matchedTP_;
  std::vector<const Stub*>   matchedStubs_;
  unsigned int          nMatchedLayers_;
  unsigned int          numStubs;
};

} // end ns vertexFinder


#endif