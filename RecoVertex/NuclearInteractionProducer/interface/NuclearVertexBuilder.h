#ifndef NuclearVertexBuilder_h_
#define NuclearVertexBuilder_h_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class FreeTrajectoryState;

class NuclearVertexBuilder {

  public :
       NuclearVertexBuilder( const MagneticField * mag ) : theMagField(mag) {}
       void build( const reco::TrackRef& primaryTrack, const reco::TrackRefVector& secondaryTrack );
       reco::Vertex  getVertex() const { return the_vertex; } 

  private :
       FreeTrajectoryState getTrajectory(const reco::TrackRef& track);
       reco::Vertex  the_vertex;
       const MagneticField * theMagField;
};

#endif
