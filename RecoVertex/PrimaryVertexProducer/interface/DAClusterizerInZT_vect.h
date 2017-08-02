#ifndef DAClusterizerInZT_vect_h
#define DAClusterizerInZT_vect_h

/**\class DAClusterizerInZT_vect

 Description: separates event tracks into clusters along the beam line

	Version which auto-vectorizes with gcc 4.6 or newer

 */

#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include "DataFormats/Math/interface/Error.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include <memory>
#include <stdlib.h>
#include <malloc.h>

template <typename T, std::size_t N = 16>
class AlignmentAllocator {
public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T * pointer;
  typedef const T * const_pointer;

  typedef T & reference;
  typedef const T & const_reference;

  public:
  inline AlignmentAllocator () throw () { }

  template <typename T2>
  inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }

  inline ~AlignmentAllocator () throw () { }

  inline pointer adress (reference r) {
    return &r;
  }

  inline const_pointer adress (const_reference r) const {
    return &r;
  }

  inline pointer allocate (size_type n) {
    void * p = nullptr;
    posix_memalign(&p,N,n*sizeof(value_type));
    return (pointer)p;
  }

  inline void deallocate (pointer p, size_type) {
    free(p);
  }

  inline void construct (pointer p, const value_type & wert) {
     new (p) value_type (wert);
  }

  inline void destroy (pointer p) {
    p->~value_type ();
  }

  inline size_type max_size () const throw () {
    return size_type (-1) / sizeof (value_type);
  }

  template <typename T2>
  struct rebind {
    typedef AlignmentAllocator<T2, N> other;
  };

  bool operator!=(const AlignmentAllocator<T,N>& other) const  {
    return !(*this == other);
  }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const AlignmentAllocator<T,N>& other) const {
    return true;
  }
};


class DAClusterizerInZT_vect  final : public TrackClusterizerInZ {

public:
  template<typename T> 
  using AlignedVector = std::vector<T, AlignmentAllocator<T, 16> >;

  // Internal data structure to 
  struct track_t {
    
    void AddItem( double new_z, double new_t, double new_dz2, double new_dt2, const reco::TransientTrack* new_tt, double new_pi   )
    {
      z.push_back( new_z );
      t.push_back( new_t );
      dz2.push_back( new_dz2 );
      dt2.push_back( new_dt2 );
      errsum.push_back( 1./(1./new_dz2 + 1./new_dt2) );
      tt.push_back( new_tt );
      
      pi.push_back( new_pi ); // track weight
      Z_sum.push_back( 1.0 ); // Z[i]   for DA clustering, initial value as done in ::fill
    }

    
    
    unsigned int GetSize() const
    {
      return z.size();
    }
    
    
    // has to be called everytime the items are modified
    void ExtractRaw()
    {
      _z = &z.front();
      _t = &t.front();
      _dz2 = &dz2.front();
      _dt2 = &dt2.front();
      _errsum = &errsum.front();
      _Z_sum = &Z_sum.front();
      _pi = &pi.front();
    }
    
    double * __restrict__ _z __attribute__ ((aligned (16))); // z-coordinate at point of closest approach to the beamline
    double * __restrict__ _t __attribute__ ((aligned (16))); // t-coordinate at point of closest approach to the beamline
    double * __restrict__ _dz2 __attribute__ ((aligned (16))); // square of the error of z(pca)
    double * __restrict__ _dt2 __attribute__ ((aligned (16))); // square of the error of t(pca)
    double * __restrict__ _errsum __attribute__ ((aligned (16))); // sum of squares of the pca errors
    
    double * __restrict__  _Z_sum __attribute__ ((aligned (16))); // Z[i]   for DA clustering
    double * __restrict__  _pi __attribute__ ((aligned (16))); // track weight
    
    AlignedVector<double> z; // z-coordinate at point of closest approach to the beamline
    AlignedVector<double> t; // t-coordinate at point of closest approach to the beamline
    AlignedVector<double> dz2; // square of the error of z(pca)
    AlignedVector<double> dt2; // square of the error of t(pca)
    AlignedVector<double> errsum; // sum of squares of the pca errors    
    AlignedVector<double> Z_sum; // Z[i]   for DA clustering
    AlignedVector<double> pi; // track weight

    std::vector< const reco::TransientTrack* > tt; // a pointer to the Transient Track
  };
  
  struct vertex_t {
    AlignedVector<double> z; //           z coordinate
    AlignedVector<double> t; //           t coordinate
    AlignedVector<double> pk; //           vertex weight for "constrained" clustering
    
    // --- temporary numbers, used during update
    AlignedVector<double> ei_cache;
    AlignedVector<double> ei;
    AlignedVector<double> sw;
    AlignedVector<double> swz;
    AlignedVector<double> swt;
    AlignedVector<double> se;
    AlignedVector<double> swE;
    
    
    unsigned int GetSize() const
    {
      return z.size();
    }
    
    void AddItem( double new_z, double new_t, double new_pk   )
    {
      z.push_back( new_z);
      t.push_back( new_t);
      pk.push_back( new_pk);
      
      ei_cache.push_back( 0.0 );
      ei.push_back( 0.0 );
      sw.push_back( 0.0 );
      swz.push_back( 0.0);
      swt.push_back( 0.0);
      se.push_back( 0.0);
      swE.push_back( 0.0);
      
      ExtractRaw();
    }
    
    void InsertItem( unsigned int i, double new_z, double new_t, double new_pk   )
    {
      z.insert(z.begin() + i, new_z);
      t.insert(t.begin() + i, new_t);
      pk.insert(pk.begin() + i, new_pk);
      
      ei_cache.insert(ei_cache.begin() + i, 0.0 );
      ei.insert( ei.begin()  + i, 0.0 );
      sw.insert( sw.begin()  + i, 0.0 );
      swz.insert(swz.begin() + i, 0.0 );
      swt.insert(swt.begin() + i, 0.0 );
      se.insert( se.begin()  + i, 0.0 );
      swE.insert(swE.begin() + i, 0.0 );
      
      ExtractRaw();
    }
    
    void RemoveItem( unsigned int i )
    {
      z.erase( z.begin() + i );
      t.erase( t.begin() + i );
      pk.erase( pk.begin() + i );
      
      ei_cache.erase( ei_cache.begin() + i);
      ei.erase( ei.begin() + i);
      sw.erase( sw.begin() + i);
      swz.erase( swz.begin() + i);
      swt.erase( swt.begin() + i);
      se.erase(se.begin() + i);
      swE.erase(swE.begin() + i);
      
      ExtractRaw();
    }
    
    void DebugOut()
    {
      std::cout <<  "vertex_t size: " << GetSize() << std::endl;
      
      for ( unsigned int i =0; i < GetSize(); ++ i)
	{
	  std::cout << " z = " << _z[i] << " t = " << _t[i] << " pk = " << _pk[i] << std::endl;
	}
    }
    
    // has to be called everytime the items are modified
    void ExtractRaw()
    {
      _z = &z.front();
      _t = &t.front();
      _pk = &pk.front();
      
      _ei = &ei.front();
      _sw = &sw.front();
      _swz = &swz.front();
      _swt = &swt.front();
      _se = &se.front();
      _swE = &swE.front();
      _ei_cache = &ei_cache.front();
      
    }
    
    double * __restrict__ _z __attribute__ ((aligned (16)));
    double * __restrict__ _t __attribute__ ((aligned (16)));
    double * __restrict__ _pk __attribute__ ((aligned (16)));
    
    double * __restrict__ _ei_cache __attribute__ ((aligned (16)));
    double * __restrict__ _ei __attribute__ ((aligned (16)));
    double * __restrict__ _sw __attribute__ ((aligned (16)));
    double * __restrict__ _swz __attribute__ ((aligned (16)));
    double * __restrict__ _swt __attribute__ ((aligned (16)));
    double * __restrict__ _se __attribute__ ((aligned (16)));
    double * __restrict__ _swE __attribute__ ((aligned (16)));
    
  };
  
  DAClusterizerInZT_vect(const edm::ParameterSet& conf);
  
  
  std::vector<std::vector<reco::TransientTrack> >
  clusterize(const std::vector<reco::TransientTrack> & tracks) const;
  
  
  std::vector<TransientVertex>
  vertices(const std::vector<reco::TransientTrack> & tracks,
	   const int verbosity = 0) const ;
  
  track_t	fill(const std::vector<reco::TransientTrack> & tracks) const;
  
  double update(double beta, track_t & gtracks,
		vertex_t & gvertices, bool useRho0, const double & rho0) const;

  void dump(const double beta, const vertex_t & y,
	    const track_t & tks, const int verbosity = 0) const;
  bool merge(vertex_t & y, double & beta)const;
  bool purge(vertex_t &, track_t &, double &,
	     const double) const;
  void splitAll( vertex_t & y) const;
  bool split(const double beta,  track_t &t, vertex_t & y, double threshold = 1. ) const;
  
  double beta0(const double betamax, track_t const & tks, vertex_t const & y) const;
    
  
private:
  bool verbose_;
  double zdumpcenter_;
  double zdumpwidth_;

  double vertexSize_;
  int maxIterations_;
  double coolingFactor_;
  double betamax_;
  double betastop_;
  double dzCutOff_;
  double d0CutOff_;
  bool useTc_;

  double mintrkweight_;
  double uniquetrkweight_;
  double zmerge_;
  double tmerge_;
  double betapurge_;

};


//#ifndef DAClusterizerInZT_new_h
#endif
