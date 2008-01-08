#ifndef HEPMC_HEPEVT_CWRAPPER_H
#define HEPMC_HEPEVT_CWRAPPER_H
//--------------------------------------------------------------------------
// This is a partial clone of HEPEVT_Wrapper, written by Matt  Dobbs
//
// This wrapper is for use with C structs when the common block is not defined
// This implementation uses the actual structure members, 
// which means you do need to know the size of the HEPEVT arrays
//
//--------------------------------------------------------------------------

#include <ctype.h>
#include <iostream>
#include <cstdio>       // needed for formatted output using sprintf 

#include "VistaTools/Stdhep/interface/stdhep.h"

extern "C" {
extern struct hepevt myhepevt;
}

static int HEPEVT_EntriesAllocation = NMXHEP;


const unsigned int hepevt_bytes_allocation = 
                sizeof(long int) * ( 2 + 4 * HEPEVT_EntriesAllocation )
                + sizeof(double) * ( 9 * HEPEVT_EntriesAllocation );


//--------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////
// Generic Wrapper for the fortran HEPEVT common block
// This class is intended for static use only - it makes no sense to 
// instantiate it.
//////////////////////////////////////////////////////////////////////////
//
// The index refers to the fortran style index: 
// i.e. index=1 refers to the first entry in the HEPEVT common block.
// all indices must be >0
// number_entries --> integer between 0 and max_number_entries() giving total
//                    number of sequential particle indices
// first_parent/child --> index of first mother/child if there is one, 
//                        zero otherwise
// last_parent/child --> if number children is >1, address of last parent/child
//                       if number of children is 1, same as first_parent/child
//                       if there are no children, returns zero.
// is_double_precision --> T or F depending if floating point variables 
//                         are 8 or 4 bytes
//


namespace HepMC {

    class HEPEVT_CWrapper {
    public:

	static void print_hepevt( std::ostream& ostr = std::cout );
	static void print_hepevt_particle( int index, 
					   std::ostream& ostr = std::cout );
        static bool is_double_precision();  // True if common block uses double

	static bool check_hepevt_consistency( std::ostream& ostr = std::cout );

	static void zero_everything();

	////////////////////
	// Access Methods //
	////////////////////
        static int    event_number();             // event number
        static int    number_entries();           // num entries in current evt
        static int    status( int index );        // status code
        static int    id( int index );            // PDG particle id
        static int    first_parent( int index );  // index of 1st mother
        static int    last_parent( int index );   // index of last mother
	static int    number_parents( int index ); 
        static int    first_child( int index );   // index of 1st daughter
        static int    last_child( int index );    // index of last daughter
	static int    number_children( int index );
        static double px( int index );            // X momentum       
        static double py( int index );
        static double pz( int index );
        static double e( int index );             // Energy
        static double m( int index );             // X Production vertex
        static double x( int index );
        static double y( int index );
        static double z( int index );
        static double t( int index );             // production time

	////////////////////
	// Set Methods    //
	////////////////////
        static void set_event_number( int evtno );
        static void set_number_entries( int noentries );
        static void set_status( int index, int status );
        static void set_id( int index, int id );
        static void set_parents( int index, int firstparent, int lastparent );
        static void set_children( int index, int firstchild, int lastchild );
        static void set_momentum( int index, double px, double py,
				  double pz, double e );
        static void set_mass( int index, double mass );
        static void set_position( int index, double x, double y, double z, 
				  double t );
	//////////////////////
	// HEPEVT Floorplan //
	//////////////////////
	static unsigned int sizeof_int();
	static unsigned int sizeof_real();
        static int  max_number_entries();
	static void set_sizeof_int(unsigned int);
	static void set_sizeof_real(unsigned int);
	static void set_max_number_entries(unsigned int);

    protected:
	static void   print_legend( std::ostream& ostr = std::cout );

    private:
	static unsigned int s_sizeof_int;
	static unsigned int s_sizeof_real;
	static unsigned int s_max_number_entries;

    }; 

    //////////////////////////////
    // HEPEVT Floorplan Inlines //
    //////////////////////////////
    inline unsigned int HEPEVT_CWrapper::sizeof_int(){ return s_sizeof_int; }

    inline unsigned int HEPEVT_CWrapper::sizeof_real(){ return s_sizeof_real; }

    inline int HEPEVT_CWrapper::max_number_entries() 
    { return (int)s_max_number_entries; }

    inline void HEPEVT_CWrapper::set_sizeof_int( unsigned int size ) 
    {
	if ( size != sizeof(short int) && size != sizeof(long int) && size != sizeof(int) ) {
	    std::cerr << "HepMC is not able to handle integers "
		      << " of size other than 2 or 4."
		      << " You requested: " << size << std::endl;
	}
	s_sizeof_int = size;
    }

    inline void HEPEVT_CWrapper::set_sizeof_real( unsigned int size ) {
	if ( size != sizeof(float) && size != sizeof(double) ) {
	    std::cerr << "HepMC is not able to handle floating point numbers"
		      << " of size other than 4 or 8."
		      << " You requested: " << size << std::endl;
	}
	s_sizeof_real = size;
    }

    inline void HEPEVT_CWrapper::set_max_number_entries( unsigned int size ) {
	s_max_number_entries = size;
    }



    //////////////
    // INLINES  //
    //////////////

    inline bool HEPEVT_CWrapper::is_double_precision() 
    { 
	// true if 8byte floating point numbers are used in the HepEVT common.
	return ( sizeof(double) == sizeof_real() );
    }

    inline int HEPEVT_CWrapper::event_number()
    { return myhepevt.nevhep; }

    inline int HEPEVT_CWrapper::number_entries() 
    { 
	int nhep = myhepevt.nhep;
	return ( nhep <= max_number_entries() ?
		 nhep : max_number_entries() );
    }

    inline int HEPEVT_CWrapper::status( int index )   
    { return myhepevt.isthep[index-1]; }

    inline int HEPEVT_CWrapper::id( int index )
    { 
	return myhepevt.idhep[index-1];
    }

    inline int HEPEVT_CWrapper::first_parent( int index )
    { 
	int parent = myhepevt.jmohep[index-1][0];
	return ( parent > 0 && parent <= number_entries() ) ?
					 parent : 0; 
    }

    inline int HEPEVT_CWrapper::last_parent( int index )
    { 
	// Returns the Index of the LAST parent in the HEPEVT record
	// for particle with Index index.
	// If there is only one parent, the last parent is forced to 
	// be the same as the first parent.
	// If there are no parents for this particle, both the first_parent
	// and the last_parent with return 0.
	// Error checking is done to ensure the parent is always
	// within range ( 0 <= parent <= nhep )
	//
	int firstparent = first_parent(index);
	int parent = myhepevt.jmohep[index-1][1];
	return ( parent > firstparent && parent <= number_entries() ) 
						   ? parent : firstparent; 
    }

    inline int HEPEVT_CWrapper::number_parents( int index ) {
	int firstparent = first_parent(index);
	return ( firstparent>0 ) ? 
	    ( 1+last_parent(index)-firstparent ) : 0;
    }

    inline int HEPEVT_CWrapper::first_child( int index )
    { 
	int child = myhepevt.jdahep[index-1][0];
	return ( child > 0 && child <= number_entries() ) ?
				       child : 0; 
    }

    inline int HEPEVT_CWrapper::last_child( int index )
    { 
	// Returns the Index of the LAST child in the HEPEVT record
	// for particle with Index index.
	// If there is only one child, the last child is forced to 
	// be the same as the first child.
	// If there are no children for this particle, both the first_child
	// and the last_child with return 0.
	// Error checking is done to ensure the child is always
	// within range ( 0 <= parent <= nhep )
	//
	int firstchild = first_child(index);
	int child = myhepevt.jdahep[index-1][1];
	return ( child > firstchild && child <= number_entries() ) 
						? child : firstchild;
    }

    inline int HEPEVT_CWrapper::number_children( int index ) 
    {
	int firstchild = first_child(index);
	return ( firstchild>0 ) ? 
	    ( 1+last_child(index)-firstchild ) : 0;
    }

    inline double HEPEVT_CWrapper::px( int index )
    { 
	return myhepevt.phep[index-1][0];
    }

    inline double HEPEVT_CWrapper::py( int index )
    { 
	return myhepevt.phep[index-1][1];
    }


    inline double HEPEVT_CWrapper::pz( int index )
    { 
	return myhepevt.phep[index-1][2];
    }

    inline double HEPEVT_CWrapper::e( int index )
    { 
	return myhepevt.phep[index-1][3];
    }

    inline double HEPEVT_CWrapper::m( int index )
    { 
	return myhepevt.phep[index-1][4];
    }

    inline double HEPEVT_CWrapper::x( int index )
    { 
	return myhepevt.vhep[index-1][0];
    }

    inline double HEPEVT_CWrapper::y( int index )
    { 
	return myhepevt.vhep[index-1][1];
    }

    inline double HEPEVT_CWrapper::z( int index )
    { 
	return myhepevt.vhep[index-1][2];
    }

    inline double HEPEVT_CWrapper::t( int index )
    { 
	return myhepevt.vhep[index-1][3];
    }

    inline void HEPEVT_CWrapper::set_event_number( int evtno ) 
    { myhepevt.nevhep = evtno; }

    inline void HEPEVT_CWrapper::set_number_entries( int noentries ) 
    { myhepevt.nhep = noentries; }

    inline void HEPEVT_CWrapper::set_status( int index, int status ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.isthep[index-1] = status;
    }

    inline void HEPEVT_CWrapper::set_id( int index, int id ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.idhep[index-1] = id;
    }

    inline void HEPEVT_CWrapper::set_parents( int index, int firstparent, 
					     int lastparent ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.jmohep[index-1][0] = firstparent;
	myhepevt.jmohep[index-1][1] = lastparent;
    }
    
    inline void HEPEVT_CWrapper::set_children( int index, int firstchild, 
					      int lastchild ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.jdahep[index-1][0] = firstchild;
	myhepevt.jdahep[index-1][1] = lastchild;
    }

    inline void HEPEVT_CWrapper::set_momentum( int index, double px, 
					      double py, double pz, double e ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.phep[index-1][0] = px;
	myhepevt.phep[index-1][1] = py;
	myhepevt.phep[index-1][2] = pz;
	myhepevt.phep[index-1][3] = e;
    }

    inline void HEPEVT_CWrapper::set_mass( int index, double mass ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.phep[index-1][4] = mass;
    }

    inline void HEPEVT_CWrapper::set_position( int index, double x, double y,
					      double z, double t ) 
    {
        if ( index <= 0 || index > max_number_entries() ) return;
	myhepevt.vhep[index-1][0] = x;
	myhepevt.vhep[index-1][1] = y;
	myhepevt.vhep[index-1][2] = z;
	myhepevt.vhep[index-1][3] = t;
    }

} // HepMC

#endif  // HEPMC_HEPEVT_CWRAPPER_H

