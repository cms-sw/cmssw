#ifndef TrimmedTrackFilter_H
#define TrimmedTrackFilter_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/** Select TransientTracks for a vertex search 
 *  with the ConfigurableTrimmedVertexFinder algorithm 
 *  by applying a pT cut. The pT cut value can be set 
 *  at runtime via the corresponding set() method. 
 *  The default value is pT > 0. GeV
 */

class TrimmedTrackFilter {

public:

  /** Constructor with cuts defined at runtime by configurables
   */
  TrimmedTrackFilter();

  ~TrimmedTrackFilter(){};

  /** Access to pT cut
   */
  double ptCut() const { return thePtCut; }

  /** Set pT cut
   */
  void setPtCut(double ptCut) { thePtCut = ptCut; }

  bool operator()(const reco::TransientTrack &) const;

private:

  double thePtCut;

};

#endif 
