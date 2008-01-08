//--------------------------------------------------------------------------
#ifndef CONVERT_HEPEVT_H
#define CONVERT_HEPEVT_H

//////////////////////////////////////////////////////////////////////////
//
// Important note: This class uses HepMC::HEPEVT_CWrapper which is an
//                 interface to the fortran77 HEPEVT common block.
//                 The precision and number of entries in the F77 common 
//                 block can be specified. See HepMC/HEPEVT_CWrapper.h.
//                 You will very likely have to specify these values for your
//                 application.
//
//

#include <set>
#include <vector>
#include "HepMC/IO_BaseClass.h"
#include "VistaTools/ConvertStdHep/interface/HEPEVT_CWrapper.h"

namespace HepMC {

    class GenEvent;
    class GenVertex;
    class GenParticle;
    class ParticleDataTable;

    class convertHEPEVT : public IO_BaseClass {
    public:
	convertHEPEVT();
	virtual           ~convertHEPEVT();
	bool              fill_next_event( GenEvent* );
	void              write_event( const GenEvent* );
	void              print( std::ostream& ostr = std::cout ) const;
	
	// see comments below for these switches.
	bool              trust_both_mothers_and_daughters() const;
	bool              trust_mothers_before_daughters() const;
	bool              print_inconsistency_errors() const;
	void              set_trust_mothers_before_daughters( bool b = 1 );
	void              set_trust_both_mothers_and_daughters( bool b = 0 );
	void              set_print_inconsistency_errors( bool b = 1 );

    protected: // for internal use only
	GenParticle* build_particle( int index );
	void build_production_vertex( 
	    int i,std::vector<GenParticle*>& hepevt_particle, GenEvent* evt );
	void build_end_vertex( 
	    int i, std::vector<GenParticle*>& hepevt_particle, GenEvent* evt );
	int  find_in_map( 
	    const std::map<GenParticle*,int>& m, GenParticle* p) const;

    private: // following are not implemented for HEPEVT
        virtual void write_particle_data_table( const ParticleDataTable* ){}
        virtual bool fill_particle_data_table( ParticleDataTable* ) 
	    { return 0; }

    private: // use of copy constructor is not allowed
	convertHEPEVT( const convertHEPEVT& ) : IO_BaseClass() {}

    private: // data members

	bool m_trust_mothers_before_daughters;
	bool m_trust_both_mothers_and_daughters;
	bool m_print_inconsistency_errors; 
	// Since HEPEVT has bi-directional pointers, it is possible that
	// the mother/daughter pointers are inconsistent (though physically
	// speaking this should never happen). In practise it happens often.
	// When a conflict occurs (i.e. when mother/daughter pointers are in 
	// disagreement, where an empty (0) pointer is not considered a 
	// disagreement) an error is printed. These errors can be turned off 
	// with:            myio_hepevt.set_print_inconsistency_errors(0);
	// but it is STRONGLY recommended that you print the HEPEVT 
	// common and understand the inconsistency BEFORE you turn off the
	// errors. The messages are there for a reason [remember, there is
	// no message printed when the information is missing, ... only when
	// is it inconsistent. User beware.]
	// You can inspect the HEPEVT common block for inconsistencies with
	//   HEPEVT_CWrapper::check_hepevt_consistency()
	//
	// There is a switch controlling whether the mother pointers or
	// the daughters are to be trusted.
	// For example, in Pythia the mother information is always correctly
	// included, but the daughter information is often left unfilled: in
	// this case we want to trust the mother pointers and not necessarily
	// the daughters. [THIS IS THE DEFAULT]. Unfortunately the reverse
	// happens for the stdhep(2001) translation of Isajet, so we need
	// an option to toggle the choices.
    };

    ////////////////////////////
    // INLINES access methods //
    ////////////////////////////
    inline bool convertHEPEVT::trust_both_mothers_and_daughters() const 
    { return m_trust_both_mothers_and_daughters; }
	
    inline bool convertHEPEVT::trust_mothers_before_daughters() const 
    { return m_trust_mothers_before_daughters; }

    inline bool convertHEPEVT::print_inconsistency_errors() const
    { return m_print_inconsistency_errors; }

    inline void convertHEPEVT::set_trust_both_mothers_and_daughters( bool b )
    { m_trust_both_mothers_and_daughters = b; }

    inline void convertHEPEVT::set_trust_mothers_before_daughters( bool b )
    { m_trust_mothers_before_daughters = b; }

    inline void convertHEPEVT::set_print_inconsistency_errors( bool b  )
    { m_print_inconsistency_errors = b; }

} // HepMC

#endif  // CONVERT_HEPEVT_H
//--------------------------------------------------------------------------
