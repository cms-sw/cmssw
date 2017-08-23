#ifndef __TMTrackTrigger_VertexFinder_L1fittedTrack_h__
#define __TMTrackTrigger_VertexFinder_L1fittedTrack_h__


#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace vertexFinder {

//! Simple wrapper class for TTTrack, to avoid changing other areas of packages immediately
class L1fittedTrack {
public:
  L1fittedTrack(const TTTrack< Ref_Phase2TrackerDigi_ >& );
  ~L1fittedTrack();

  float eta() const;
  float phi0() const;
  float pt() const;
  float z0() const;

private:
  TTTrack< Ref_Phase2TrackerDigi_ > track_;
};

} // end ns vertexFinder


#endif