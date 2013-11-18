#ifndef __L1Analysis_L1AnalysisRecoTrack_H__
#define __L1Analysis_L1AnalysisRecoTrack_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1TrackVertexRecoTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "L1AnalysisRecoTrackDataFormat.h"


namespace L1Analysis
{
  class L1AnalysisRecoTrack 
  {
  public:
    L1AnalysisRecoTrack(){Reset();};
    ~L1AnalysisRecoTrack(){};
    
    void SetTracks(const reco::TrackCollection& trackColl, unsigned maxTrack);
    L1Analysis::L1AnalysisRecoTrackDataFormat * getData(){return (&track_);}
    void Reset() {track_.Reset();}

  private :            
    L1Analysis::L1AnalysisRecoTrackDataFormat track_;
  }; 
}
#endif


