#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include <CLHEP/HepMC/GenParticle.h>
#include <CLHEP/HepMC/GenVertex.h>

TrackingParticle::TrackingParticle(  const HepMC::GenParticle * p ) :
  reco::Particle( int( p->particledata().charge() ), 
		  LorentzVector( p->momentum().x(), 
				 p->momentum().y(), 
				 p->momentum().z(), 
				 p->momentum().t() ), 
		  Point( p->production_vertex()->point3d().x(),
			 p->production_vertex()->point3d().y(),
			 p->production_vertex()->point3d().z() )
		  ),
  genParticle_( p ), pdgId_( p->pdg_id() ) {
}
