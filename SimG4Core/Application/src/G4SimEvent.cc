#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include "G4SystemOfUnits.hh"

class IdSort{
public:
  bool operator()(const SimTrack& a, const SimTrack& b) {
    return a.trackId() < b.trackId();
  }
};


G4SimEvent::G4SimEvent() : hepMCEvent(0),
                           weight_(0),
                           collisionPoint_(math::XYZTLorentzVectorD(0.,0.,0.,0.)),
			   nparam_(0),param_(0) {}

G4SimEvent::~G4SimEvent() 
{

/*
   while ( !g4tracks.empty() )
   {
      delete g4tracks.back() ;
      g4tracks.pop_back() ;
   }
   while ( !g4vertices.empty() )
   {
      delete g4vertices.back() ;
      g4vertices.pop_back() ;
   }
*/

   // per suggestion by Chris Jones, it's faster 
   // that delete back() and pop_back() 
   //
   unsigned int i = 0 ;
   
   for ( i=0; i<g4tracks.size(); i++ )
   {
      delete g4tracks[i] ;
      g4tracks[i] = 0 ;
   }
   g4tracks.clear() ;
   
   for ( i=0; i<g4vertices.size(); i++ )
   {
      delete g4vertices[i] ;
      g4vertices[i] = 0 ;
   }
   g4vertices.clear();
}

void G4SimEvent::load(edm::SimTrackContainer & c) const
{
    for (unsigned int i=0; i<g4tracks.size(); i++)
    {
	G4SimTrack * trk    = g4tracks[i];
	int ip              = trk->part();
	math::XYZTLorentzVectorD p( trk->momentum().x()/GeV,
	                            trk->momentum().y()/GeV,
                                    trk->momentum().z()/GeV,
			            trk->energy()/GeV ) ;
	int iv              = trk->ivert();
	int ig              = trk->igenpart();
	int id              = trk->id();
	math::XYZVectorD tkpos( trk->trackerSurfacePosition().x()/cm,
	                        trk->trackerSurfacePosition().y()/cm,
			        trk->trackerSurfacePosition().z()/cm ) ;
	math::XYZTLorentzVectorD tkmom( trk->trackerSurfaceMomentum().x()/GeV,
	                                trk->trackerSurfaceMomentum().y()/GeV,
	                                trk->trackerSurfaceMomentum().z()/GeV,
				        trk->trackerSurfaceMomentum().e()/GeV ) ;
	// ip = particle ID as PDG
	// pp = 4-momentum
	// iv = corresponding G4SimVertex index
	// ig = corresponding GenParticle index
	SimTrack t = SimTrack(ip,p,iv,ig,tkpos,tkmom);
	t.setTrackId(id);
	t.setEventId(EncodedEventId(0));
	c.push_back(t);
    }
    std::stable_sort(c.begin(),c.end(),IdSort());
    
}

void G4SimEvent::load(edm::SimVertexContainer & c) const
{
    for (unsigned int i=0; i<g4vertices.size(); i++)
    {
	G4SimVertex * vtx   = g4vertices[i];
	//
	// starting 1_1_0_pre3, SimVertex stores in cm !!!
	// 
	math::XYZVectorD v3( vtx->vertexPosition().x()/cm, 
	                     vtx->vertexPosition().y()/cm,
			     vtx->vertexPosition().z()/cm ) ;
	float t             = vtx->vertexGlobalTime()/second;
	int iv              = vtx->parentIndex();
	// vv = position
	// t  = global time
	// iv = index of the parent in the SimEvent SimTrack container (-1 if no parent)
	SimVertex v = SimVertex(v3,t,iv,i);
        v.setProcessType(vtx->processType());
	v.setEventId(EncodedEventId(0));
	c.push_back(v);
    }
}

