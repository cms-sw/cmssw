//////////////////////////////////////////////////////////////////////////
// This is based on IO_HEPEVT
// HEPEVT IO class
//////////////////////////////////////////////////////////////////////////

#include "VistaTools/ConvertStdHep/interface/convertHEPEVT.h"
#include "HepMC/GenEvent.h"
#include <cstdio>       // needed for formatted output using sprintf 

extern "C" {
extern struct hepevt myhepevt;
}

namespace HepMC {

    convertHEPEVT::convertHEPEVT() : m_trust_mothers_before_daughters(1),
			     m_trust_both_mothers_and_daughters(0),
			     m_print_inconsistency_errors(1)
    {}

    convertHEPEVT::~convertHEPEVT(){}

    void convertHEPEVT::print( std::ostream& ostr ) const { 
        ostr << "convertHEPEVT: reads an event from the FORTRAN HEPEVT "
             << "common block. \n" 
	     << " trust_mothers_before_daughters = " 
	     << m_trust_mothers_before_daughters
	     << " trust_both_mothers_and_daughters = "
	     << m_trust_both_mothers_and_daughters
	     << ", print_inconsistency_errors = " 
	     << m_print_inconsistency_errors << std::endl;
    }

    bool convertHEPEVT::fill_next_event( GenEvent* evt ) {
	// read one event from the HEPEVT common block and fill GenEvent
	// return T/F =success/failure
	//
	// For HEPEVT commons built with the luhepc routine of Pythia 5.7
	//  the children pointers are not always correct (i.e. there is 
	//  oftentimes an internal inconsistency between the parents and 
	//  children pointers). The parent pointers always seem to be correct.
	// Thus the switch trust_mothers_before_daughters=1 is appropriate for
	//  pythia. NOTE: you should also set the switch MSTP(128) = 2 in 
	//                pythia (not the default!), so that pythia doesn't
	//                store two copies of resonances in the event record.
	// The situation is opposite for the HEPEVT which comes from Isajet
	// via stdhep, so then use the switch trust_mothers_before_daughters=0
	//
	// 1. test that evt pointer is not null and set event number
	if ( !evt ) {
	    std::cerr 
		<< "convertHEPEVT::fill_next_event error - passed null event." 
		<< std::endl;
	    return 0;
	}
	evt->set_event_number( HEPEVT_CWrapper::event_number() );
	//
	// 2. create a particle instance for each HEPEVT entry and fill a map
	//    create a vector which maps from the HEPEVT particle index to the 
	//    GenParticle address
	//    (+1 in size accounts for hepevt_particle[0] which is unfilled)
	std::vector<GenParticle*> hepevt_particle( 
	                                HEPEVT_CWrapper::number_entries()+1 );
	hepevt_particle[0] = 0;
	for ( int i1 = 1; i1 <= HEPEVT_CWrapper::number_entries(); ++i1 ) {
	    hepevt_particle[i1] = build_particle(i1);
	}
	std::set<GenVertex*> new_vertices;
	//
	// Here we assume that the first two particles in the list 
	// are the incoming beam particles.
	evt->set_beam_particles( hepevt_particle[1], hepevt_particle[2] );
	//
	// 3.+4. loop over HEPEVT particles AGAIN, this time creating vertices
	for ( int i = 1; i <= HEPEVT_CWrapper::number_entries(); ++i ) {
	    // We go through and build EITHER the production or decay 
	    // vertex for each entry in hepevt, depending on the switch
	    // m_trust_mothers_before_daughters (new 2001-02-28)
	    // Note: since the HEPEVT pointers are bi-directional, it is
	    //      sufficient to do one or the other.
	    //
	    // 3. Build the production_vertex (if necessary)
	    if ( m_trust_mothers_before_daughters || 
		 m_trust_both_mothers_and_daughters ) {
		build_production_vertex( i, hepevt_particle, evt );
	    }
	    //
	    // 4. Build the end_vertex (if necessary) 
	    //    Identical steps as for production vertex
	    if ( !m_trust_mothers_before_daughters || 
		 m_trust_both_mothers_and_daughters ) {
		build_end_vertex( i, hepevt_particle, evt );
	    }
	}
	// 5.             01.02.2000
	// handle the case of particles in HEPEVT which come from nowhere -
	//  i.e. particles without mothers or daughters.
	//  These particles need to be attached to a vertex, or else they
	//  will never become part of the event. check for this situation
	for ( int i3 = 1; i3 <= HEPEVT_CWrapper::number_entries(); ++i3 ) {
	    if ( !hepevt_particle[i3]->end_vertex() && 
			!hepevt_particle[i3]->production_vertex() ) {
		GenVertex* prod_vtx = new GenVertex();
		prod_vtx->add_particle_out( hepevt_particle[i3] );
		evt->add_vertex( prod_vtx );
	    }
	}
	return 1;
    }

    void convertHEPEVT::write_event( const GenEvent* evt ) {
	// This writes an event out to the HEPEVT common block. The daughters
	// field is NOT filled, because it is possible to contruct graphs
	// for which the mothers and daughters cannot both be make sequential.
	// This is consistent with how pythia fills HEPEVT (daughters are not
	// necessarily filled properly) and how convertHEPEVT reads HEPEVT.
	//
	if ( !evt ) return;
	//
	// map all particles onto a unique index
	std::vector<GenParticle*> index_to_particle(
	    HEPEVT_CWrapper::max_number_entries()+1 );
	index_to_particle[0]=0;
	std::map<GenParticle*,int> particle_to_index;
	int particle_counter=0;
	for ( GenEvent::vertex_const_iterator v = evt->vertices_begin();
	      v != evt->vertices_end(); ++v ) {
	    // all "mothers" or particles_in are kept adjacent in the list
	    // so that the mother indices in hepevt can be filled properly
	    for ( GenVertex::particles_in_const_iterator p1 
		      = (*v)->particles_in_const_begin();
		  p1 != (*v)->particles_in_const_end(); ++p1 ) {
		++particle_counter;
		if ( particle_counter > 
		     HEPEVT_CWrapper::max_number_entries() ) break; 
		index_to_particle[particle_counter] = *p1;
		particle_to_index[*p1] = particle_counter;
	    }
	    // daughters are entered only if they aren't a mother of 
	    // another vtx
	    for ( GenVertex::particles_out_const_iterator p2 
		      = (*v)->particles_out_const_begin();
		  p2 != (*v)->particles_out_const_end(); ++p2 ) {
		if ( !(*p2)->end_vertex() ) {
		    ++particle_counter;
		    if ( particle_counter > 
			 HEPEVT_CWrapper::max_number_entries() ) {
			break;
		    }
		    index_to_particle[particle_counter] = *p2;
		    particle_to_index[*p2] = particle_counter;
		}
	    }
	}
	if ( particle_counter > HEPEVT_CWrapper::max_number_entries() ) {
	    particle_counter = HEPEVT_CWrapper::max_number_entries();
	}
	// 	
	// fill the HEPEVT event record
	HEPEVT_CWrapper::set_event_number( evt->event_number() );
	HEPEVT_CWrapper::set_number_entries( particle_counter );
	for ( int i = 1; i <= particle_counter; ++i ) {
	    HEPEVT_CWrapper::set_status( i, index_to_particle[i]->status() );
	    HEPEVT_CWrapper::set_id( i, index_to_particle[i]->pdg_id() );
	    FourVector m = index_to_particle[i]->momentum();
	    HEPEVT_CWrapper::set_momentum( i, m.px(), m.py(), m.pz(), m.e() );
	    HEPEVT_CWrapper::set_mass( i, index_to_particle[i]->generatedMass() );
	    if ( index_to_particle[i]->production_vertex() ) {
		FourVector p = index_to_particle[i]->
				     production_vertex()->position();
		HEPEVT_CWrapper::set_position( i, p.x(), p.y(), p.z(), p.t() );
		int num_mothers = index_to_particle[i]->production_vertex()->
				  particles_in_size();
		int first_mother = find_in_map( particle_to_index,
						*(index_to_particle[i]->
						  production_vertex()->
						  particles_in_const_begin()));
		int last_mother = first_mother + num_mothers - 1;
		if ( first_mother == 0 ) last_mother = 0;
		HEPEVT_CWrapper::set_parents( i, first_mother, last_mother );
	    } else {
		HEPEVT_CWrapper::set_position( i, 0, 0, 0, 0 );
		HEPEVT_CWrapper::set_parents( i, 0, 0 );
	    }
	    HEPEVT_CWrapper::set_children( i, 0, 0 );
	}
    }

    void convertHEPEVT::build_production_vertex(int i, 
					    std::vector<GenParticle*>& 
					    hepevt_particle,
					    GenEvent* evt ) {
	// 
	// for particle in HEPEVT with index i, build a production vertex
	// if appropriate, and add that vertex to the event
	GenParticle* p = hepevt_particle[i];
	// a. search to see if a production vertex already exists
	int mother = HEPEVT_CWrapper::first_parent(i);
	GenVertex* prod_vtx = p->production_vertex();
	while ( !prod_vtx && mother > 0 ) {
	    prod_vtx = hepevt_particle[mother]->end_vertex();
	    if ( prod_vtx ) prod_vtx->add_particle_out( p );
	    // increment mother for next iteration
	    if ( ++mother > HEPEVT_CWrapper::last_parent(i) ) mother = 0;
	}
	// b. if no suitable production vertex exists - and the particle
	// has atleast one mother or position information to store - 
	// make one
	FourVector prod_pos( HEPEVT_CWrapper::x(i), HEPEVT_CWrapper::y(i), 
				   HEPEVT_CWrapper::z(i), HEPEVT_CWrapper::t(i) 
	                         ); 
	if ( !prod_vtx && (HEPEVT_CWrapper::number_parents(i)>0 
			   || prod_pos!=FourVector(0,0,0,0)) )
	{
	    prod_vtx = new GenVertex();
	    prod_vtx->add_particle_out( p );
	    evt->add_vertex( prod_vtx );
	}
	// c. if prod_vtx doesn't already have position specified, fill it
	if ( prod_vtx && prod_vtx->position()==FourVector(0,0,0,0) ) {
	    prod_vtx->set_position( prod_pos );
	}
	// d. loop over mothers to make sure their end_vertices are
	//     consistent
	mother = HEPEVT_CWrapper::first_parent(i);
	while ( prod_vtx && mother > 0 ) {
	    if ( !hepevt_particle[mother]->end_vertex() ) {
		// if end vertex of the mother isn't specified, do it now
		prod_vtx->add_particle_in( hepevt_particle[mother] );
	    } else if (hepevt_particle[mother]->end_vertex() != prod_vtx ) {
		// problem scenario --- the mother already has a decay
		// vertex which differs from the daughter's produciton 
		// vertex. This means there is internal
		// inconsistency in the HEPEVT event record. Print an
		// error
		// Note: we could provide a fix by joining the two 
		//       vertices with a dummy particle if the problem
		//       arrises often with any particular generator.
		if ( m_print_inconsistency_errors ) std::cerr
		    << "HepMC::convertHEPEVT: inconsistent mother/daugher "
		    << "information in HEPEVT event " 
		    << HEPEVT_CWrapper::event_number()
		    << ". \n I recommend you try "
		    << "inspecting the event first with "
		    << "\n\tHEPEVT_CWrapper::check_hepevt_consistency()"
		    << "\n This warning can be turned off with the "
		    << "convertHEPEVT::print_inconsistency_errors switch."
		    << std::endl;
	    }
	    if ( ++mother > HEPEVT_CWrapper::last_parent(i) ) mother = 0;
	}
    }

    void convertHEPEVT::build_end_vertex
    ( int i, std::vector<GenParticle*>& hepevt_particle, GenEvent* evt ) 
    {
	// 
	// for particle in HEPEVT with index i, build an end vertex
	// if appropriate, and add that vertex to the event
	//    Identical steps as for build_production_vertex
	GenParticle* p = hepevt_particle[i];
	// a.
	int daughter = HEPEVT_CWrapper::first_child(i);
	GenVertex* end_vtx = p->end_vertex();
	while ( !end_vtx && daughter > 0 ) {
	    end_vtx = hepevt_particle[daughter]->production_vertex();
	    if ( end_vtx ) end_vtx->add_particle_in( p );
	    if ( ++daughter > HEPEVT_CWrapper::last_child(i) ) daughter = 0;
	}
	// b. (different from 3c. because HEPEVT particle can not know its
	//        decay position )
	if ( !end_vtx && HEPEVT_CWrapper::number_children(i)>0 ) {
	    end_vtx = new GenVertex();
	    end_vtx->add_particle_in( p );
	    evt->add_vertex( end_vtx );
	}
	// c+d. loop over daughters to make sure their production vertices 
	//    point back to the current vertex.
	//    We get the vertex position from the daughter as well.
	daughter = HEPEVT_CWrapper::first_child(i);
	while ( end_vtx && daughter > 0 ) {
	    if ( !hepevt_particle[daughter]->production_vertex() ) {
		// if end vertex of the mother isn't specified, do it now
		end_vtx->add_particle_out( hepevt_particle[daughter] );
		// 
		// 2001-03-29 M.Dobbs, fill vertex the position.
		if ( end_vtx->position()==FourVector(0,0,0,0) ) {
		    FourVector prod_pos( HEPEVT_CWrapper::x(daughter), 
					       HEPEVT_CWrapper::y(daughter), 
					       HEPEVT_CWrapper::z(daughter), 
					       HEPEVT_CWrapper::t(daughter) 
			);
		    if ( prod_pos != FourVector(0,0,0,0) ) {
			end_vtx->set_position( prod_pos );
		    }
		}
	    } else if (hepevt_particle[daughter]->production_vertex() 
		       != end_vtx){
		// problem scenario --- the daughter already has a prod
		// vertex which differs from the mother's end 
		// vertex. This means there is internal
		// inconsistency in the HEPEVT event record. Print an
		// error
		if ( m_print_inconsistency_errors ) std::cerr
		    << "HepMC::convertHEPEVT: inconsistent mother/daugher "
		    << "information in HEPEVT event " 
		    << HEPEVT_CWrapper::event_number()
		    << ". \n I recommend you try "
		    << "inspecting the event first with "
		    << "\n\tHEPEVT_CWrapper::check_hepevt_consistency()"
		    << "\n This warning can be turned off with the "
		    << "convertHEPEVT::print_inconsistency_errors switch."
		    << std::endl;
	    }
	    if ( ++daughter > HEPEVT_CWrapper::last_child(i) ) daughter = 0;
	}
	if ( !p->end_vertex() && !p->production_vertex() ) {
	    // Added 2001-11-04, to try and handle Isajet problems.
	    build_production_vertex( i, hepevt_particle, evt );
	}
    }

    GenParticle* convertHEPEVT::build_particle( int index ) {
	// Builds a particle object corresponding to index in HEPEVT
	// 
	GenParticle* p 
	    = new GenParticle( FourVector( HEPEVT_CWrapper::px(index), 
						 HEPEVT_CWrapper::py(index), 
						 HEPEVT_CWrapper::pz(index), 
						 HEPEVT_CWrapper::e(index) ),
			       HEPEVT_CWrapper::id(index), 
			       HEPEVT_CWrapper::status(index) );
        p->setGeneratedMass( HEPEVT_CWrapper::m(index) );
	p->suggest_barcode( index );
	return p;
    }

    int convertHEPEVT::find_in_map( const std::map<GenParticle*,int>& m, 
				GenParticle* p) const {
        std::map<GenParticle*,int>::const_iterator iter = m.find(p);
        if ( iter == m.end() ) return 0;
        return iter->second;
    }

} // HepMC



