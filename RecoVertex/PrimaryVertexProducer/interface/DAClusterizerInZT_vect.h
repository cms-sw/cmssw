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

class DAClusterizerInZT_vect  final : public TrackClusterizerInZ {

public:
  
  // Internal data structure to 
  struct track_t {
    
    void addItem( double new_z, double new_t, double new_dz2, double new_dt2, const reco::TransientTrack* new_tt, double new_pi   )
    {
      z.push_back( new_z );
      t.push_back( new_t );
      dz2.push_back( new_dz2 );
      dt2.push_back( new_dt2 );
      tt.push_back( new_tt );
      
      pi.push_back( new_pi ); // track weight
      Z_sum.push_back( 1.0 ); // Z[i]   for DA clustering, initial value as done in ::fill
    }
    
    unsigned int getSize() const
    {
      return z.size();
    }
    
    // has to be called everytime the items are modified
    void extractRaw()
    {
      z_ = &z.front();
      t_ = &t.front();
      dz2_ = &dz2.front();
      dt2_ = &dt2.front();
      Z_sum_ = &Z_sum.front();
      pi_ = &pi.front();
    }
    
    double * z_; // z-coordinate at point of closest approach to the beamline
    double * t_; // t-coordinate at point of closest approach to the beamline
    double * pi_; // track weight

    double * dz2_; // square of the error of z(pca)
    double * dt2_; // square of the error of t(pca)
    // double * errsum_; // sum of squares of the pca errors     obsolete
    double * Z_sum_; // Z[i]   for DA clustering
    
    std::vector<double> z; // z-coordinate at point of closest approach to the beamline
    std::vector<double> t; // t-coordinate at point of closest approach to the beamline
    std::vector<double> dz2; // square of the error of z(pca)
    std::vector<double> dt2; // square of the error of t(pca)
    //std::vector<double> errsum; // sum of squares of the pca errors    
    std::vector<double> Z_sum; // Z[i]   for DA clustering
    std::vector<double> pi; // track weight
    std::vector< const reco::TransientTrack* > tt; // a pointer to the Transient Track
  };
  
  struct vertex_t {
    
    void addItem( double new_z, double new_t, double new_pk   )
    {
      z.push_back( new_z);
      t.push_back( new_t);
      pk.push_back( new_pk);
      
      ei_cache.push_back( 0.0 );
      ei.push_back( 0.0 );
      //sw.push_back( 0.0 );
      swz.push_back( 0.0);
      swt.push_back( 0.0);
      se.push_back( 0.0);
      //swE.push_back( 0.0);
      nuz.push_back(0.0);
      nut.push_back(0.0);
      szz.push_back(0.0);
      stt.push_back(0.0);
      szt.push_back(0.0);
      
      extractRaw();
    }
    
    unsigned int getSize() const
    {
      return z.size();
    }

    // has to be called everytime the items are modified
    void extractRaw()
    {
      z_ = &z.front();
      t_ = &t.front();
      pk_ = &pk.front();
      
      ei_ = &ei.front();
      //sw_ = &sw.front();
      swz_ = &swz.front();
      swt_ = &swt.front();
      se_ = &se.front();
      //swE_ = &swE.front();
      nuz_ = &nuz.front();
      nut_ = &nut.front();
      szz_ = &szz.front();
      stt_ = &stt.front();
      szt_ = &szt.front();

      ei_cache_ = &ei_cache.front();
      
    }

    void insertItem( unsigned int i, double new_z, double new_t, double new_pk   )
    {
      z.insert(z.begin() + i, new_z);
      t.insert(t.begin() + i, new_t);
      pk.insert(pk.begin() + i, new_pk);
      
      ei_cache.insert(ei_cache.begin() + i, 0.0 );
      ei.insert( ei.begin()  + i, 0.0 );
      swz.insert(swz.begin() + i, 0.0 );
      swt.insert(swt.begin() + i, 0.0 );
      se.insert( se.begin()  + i, 0.0 );

      nuz.insert(nuz.begin() +i, 0.0 );
      nut.insert(nut.begin() +i, 0.0 );
      szz.insert(szz.begin() + i, 0.0 );
      stt.insert(stt.begin() + i, 0.0 );
      szt.insert(szt.begin() + i, 0.0 );
      extractRaw();
    }
    
    void removeItem( unsigned int i )
    {
      z.erase( z.begin() + i );
      t.erase( t.begin() + i );
      pk.erase( pk.begin() + i );
      
      ei_cache.erase( ei_cache.begin() + i);
      ei.erase( ei.begin() + i);
      swz.erase( swz.begin() + i);
      swt.erase( swt.begin() + i);
      se.erase(se.begin() + i);

      nuz.erase(nuz.begin() + i);
      nut.erase(nut.begin() + i);
      szz.erase(szz.begin() + i);
      stt.erase(stt.begin() + i);
      szt.erase(szt.begin() + i);
      
      extractRaw();
    }
    
 
    unsigned int insertOrdered( double z, double t, double pk){
      // insert a new cluster according to it's z-position, return the index at which it was inserted

      unsigned int k = 0;
      for( ; k < getSize(); k++){
	if (z < z_[k]) break;
      }
      insertItem(k ,z, t, pk);
      return k;
    }


   void debugOut()
    {
      std::cout <<  "vertex_t size: " << getSize() << std::endl;
      
      for ( unsigned int i =0; i < getSize(); ++ i)
	{
	  std::cout << " z = " << z_[i] << " t = " << t_[i] << " pk = " << pk_[i] << std::endl;
	}
    }
    
    std::vector<double> z;  //           z coordinate
    std::vector<double> t;  //           t coordinate
    std::vector<double> pk; //          vertex weight for "constrained" clustering
    
    double * z_;
    double * t_;
    double * pk_;
    
    double * ei_cache_;
    double * ei_;
    //double * sw_;
    double * swz_;
    double * swt_;
    double * se_;
    //    double * swE_;   
    double * szz_;
    double * stt_;
    double * szt_;
    double * nuz_;
    double * nut_;
    
    // --- temporary numbers, used during update
    std::vector<double> ei_cache;
    std::vector<double> ei;
    //std::vector<double> sw; replaced by nuz, nut
    std::vector<double> swz;
    std::vector<double> swt;
    std::vector<double> se;
    //    std::vector<double> swE; replaced by szz,stt,stz
    std::vector<double> nuz;
    std::vector<double> nut;
    std::vector<double> szz;
    std::vector<double> stt;
    std::vector<double> szt;
  };
  
  DAClusterizerInZT_vect(const edm::ParameterSet& conf);  
  
  std::vector<std::vector<reco::TransientTrack> >
  clusterize(const std::vector<reco::TransientTrack> & tracks) const override;  
  
  std::vector<TransientVertex>
  vertices(const std::vector<reco::TransientTrack> & tracks,
	   const int verbosity = 0) const ;
  
  track_t	fill(const std::vector<reco::TransientTrack> & tracks) const;
  
  double update(double beta, track_t & gtracks,
		vertex_t & gvertices, bool useRho0, const double & rho0) const;

  void dump(const double beta, const vertex_t & y,
	    const track_t & tks, const int verbosity = 0) const;
  void zorder(vertex_t & y)const;
  bool find_nearest(double z, double t, vertex_t & y, unsigned int & k_min, double dz, double dt)const;
  bool merge(vertex_t & y, double & beta)const;
  bool purge(vertex_t &, track_t &, double &,
	     const double) const;
  void splitAll( vertex_t & y) const;
  bool split(const double beta,  track_t &t, vertex_t & y, double threshold = 1. ) const;
  
  double beta0(const double betamax, track_t const & tks, vertex_t const & y) const;
    
  double get_Tc(const vertex_t & y, int k) const;
  
private:
  bool verbose_;
  double zdumpcenter_;
  double zdumpwidth_;

  double vertexSize_;
  double vertexSizeTime_;
  int maxIterations_;
  double coolingFactor_;
  double betamax_;
  double betastop_;
  double dzCutOff_;
  double d0CutOff_;
  double dtCutOff_;
  bool useTc_;

  double mintrkweight_;
  double uniquetrkweight_;
  double zmerge_;
  double tmerge_;
  double betapurge_;

};


//#ifndef DAClusterizerInZT_new_h
#endif
