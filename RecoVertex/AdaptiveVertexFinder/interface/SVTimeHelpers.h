#ifndef __RecoVertex_AdaptiveVertexFinder_SVTimeHelpers_h__
#define __RecoVertex_AdaptiveVertexFinder_SVTimeHelpers_h__ 

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "FWCore/Utilities/interface/isFinite.h"

namespace svtime{

  inline void updateVertexTime(TransientVertex& vtx) {
    const auto& trks = vtx.originalTracks();
    double meantime = 0., expv_x2 = 0., normw = 0., timecov = 0.;		  
    for( const auto& trk : trks ) {
      if( edm::isFinite(trk.timeExt()) ) {
	const double time = trk.timeExt();
	const double inverr = 1.0/trk.dtErrorExt();
	const double w = inverr*inverr;
	meantime += time*w;
	expv_x2  += time*time*w;
	normw    += w;
      }
    }
    if( normw > 0. ) {
      meantime = meantime/normw;
      expv_x2 = expv_x2/normw;
      timecov = expv_x2 - meantime*meantime;
      auto err = vtx.positionError().matrix4D();
      err(3,3) = timecov/(double)trks.size();  
      vtx = TransientVertex(vtx.position(),meantime,err,vtx.originalTracks(),vtx.totalChiSquared());
      //std::cout << "updated svertex time: " << vtx.time() << " +/- " << std::sqrt(err(3,3)) << std::endl;
    }
  }
}

#endif
