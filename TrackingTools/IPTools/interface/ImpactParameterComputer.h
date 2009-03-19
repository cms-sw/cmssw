#ifndef IMPACTPARAMETERCOMPUTER_H
#define IMPACTPARAMETERCOMPUTER_H

/** \class ImpactParameterComputer
 *
 * uses the track of a particle (e.g. muon, electron) and
 * a primary vertex or a given BeamSpot to calculate
 * the ImpactParameter and its error
 * \author Jasmin Kiefer
 *
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"


namespace IPTools{

  class ImpactParameterComputer{

  public:
    ImpactParameterComputer(const reco::Vertex   vtx);
    ImpactParameterComputer(const reco::BeamSpot bsp);
    ~ImpactParameterComputer();
    
    Measurement1D computeIP(const edm::EventSetup& es, const reco::Track tr, bool return3D=false);
    Measurement1D computeIPdz(const edm::EventSetup& es, const reco::Track tr);
    
  private:
    
    GlobalPoint _VtxPos;
    GlobalError _VtxPosErr;
    
  };
  
}

#endif

