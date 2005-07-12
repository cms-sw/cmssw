#include "SimG4Core/Application/interface/G4SimEvent.h"

G4SimEvent::G4SimEvent() : hepMCEvent(0),weight_(0),collisionPoint_(0),
			   nparam_(0),param_(0) {}

G4SimEvent::~G4SimEvent() {}

void G4SimEvent::load(edm::EmbdSimTrackContainer & c) const
{
    for (unsigned int i=0; i<g4tracks.size(); i++)
    {
	G4SimTrack * trk    = g4tracks[i];
	int ip              = trk->part();
	HepLorentzVector p  = HepLorentzVector(trk->momentum()/GeV,trk->energy()/GeV);
	int iv              = trk->ivert();
	int ig              = trk->igenpart();
	// ip = particle ID as PDG
	// pp = 4-momentum
	// iv = corresponding G4SimVertex index
	// ig = corresponding GenParticle index
	EmbdSimTrack t = EmbdSimTrack(ip,p,iv,ig);
	c.insertTrack(t);
    }
}

void G4SimEvent::load(edm::EmbdSimVertexContainer & c) const
{
    for (unsigned int i=0; i<g4vertices.size(); i++)
    {
	G4SimVertex * vtx   = g4vertices[i];
	Hep3Vector v3       = vtx->vertexPosition();
	float t             = vtx->vertexGlobalTime()/second;
	int iv              = vtx->parentIndex();
	// vv = position
	// t  = global time
	// iv = index of the parent in the SimEvent SimTrack container (-1 if no parent)
	EmbdSimVertex v = EmbdSimVertex(v3,t,iv);
	c.insertVertex(v);
    }
}
