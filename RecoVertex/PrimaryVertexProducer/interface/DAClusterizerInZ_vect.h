#ifndef DAClusterizerInZ_vect_h
#define DAClusterizerInZ_vect_h

/**\class DAClusterizerInZ_vect

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


class DAClusterizerInZ_vect: public TrackClusterizerInZ {

public:
	// Internal data structure to 
	struct track_t {

		void AddItem( double new_z, double new_dz2, const reco::TransientTrack* new_tt, double new_pi   )
		{
			z.push_back( new_z );
			dz2.push_back( new_dz2 );
			tt.push_back( new_tt );

			pi.push_back( new_pi ); // track weight
			Z_sum.push_back( 1.0); // Z[i]   for DA clustering, initial value as done in ::fill
		}



		unsigned int GetSize() const
		{
			return z.size();
		}


		// has to be called everytime the items are modified
		void ExtractRaw()
		{
			_z = &z.front();
			_dz2 = &dz2.front();
			_Z_sum = &Z_sum.front();
			_pi = &pi.front();
		}

		double * __restrict__ _z; // z-coordinate at point of closest approach to the beamline
		double * __restrict__  _dz2; // square of the error of z(pca)

		double * __restrict__  _Z_sum; // Z[i]   for DA clustering
		double * __restrict__  _pi; // track weight

		std::vector<double> z; // z-coordinate at point of closest approach to the beamline
		std::vector<double> dz2; // square of the error of z(pca)
		std::vector< const reco::TransientTrack* > tt; // a pointer to the Transient Track

		std::vector<double> Z_sum; // Z[i]   for DA clustering
		std::vector<double> pi; // track weight
	};

	struct vertex_t {
		std::vector<double> z; //           z coordinate
		std::vector<double> pk; //           vertex weight for "constrained" clustering

		// --- temporary numbers, used during update
		std::vector<double> ei_cache;
		std::vector<double> ei;
		std::vector<double> sw;
		std::vector<double> swz;
		std::vector<double> se;
		std::vector<double> swE;


		unsigned int GetSize() const
		{
			return z.size();
		}

		void AddItem( double new_z, double new_pk   )
		{
			z.push_back( new_z);
			pk.push_back( new_pk);

			ei_cache.push_back( 0.0 );
			ei.push_back( 0.0 );
			sw.push_back( 0.0 );
			swz.push_back( 0.0);
			se.push_back( 0.0);
			swE.push_back( 0.0);

			ExtractRaw();
		}

	        void InsertItem( unsigned int i, double new_z, double new_pk   )
		{
		        z.insert(z.begin() + i, new_z);
			pk.insert(pk.begin() + i, new_pk);

			ei_cache.insert(ei_cache.begin() + i, 0.0 );
			ei.insert( ei.begin()  + i, 0.0 );
			sw.insert( sw.begin()  + i, 0.0 );
			swz.insert(swz.begin() + i, 0.0 );
			se.insert( se.begin()  + i, 0.0 );
			swE.insert(swE.begin() + i, 0.0 );

			ExtractRaw();
		}

		void RemoveItem( unsigned int i )
		{
			z.erase( z.begin() + i );
			pk.erase( pk.begin() + i );

			ei_cache.erase( ei_cache.begin() + i);
			ei.erase( ei.begin() + i);
			sw.erase( sw.begin() + i);
			swz.erase( swz.begin() + i);
			se.erase(se.begin() + i);
			swE.erase(swE.begin() + i);

			ExtractRaw();
		}

		void DebugOut()
		{
			std::cout <<  "vertex_t size: " << GetSize() << std::endl;

			for ( unsigned int i =0; i < GetSize(); ++ i)
			{
				std::cout << " z = " << _z[i] << " pk = " << _pk[i] << std::endl;
			}
		}

		// has to be called everytime the items are modified
		void ExtractRaw()
		{
			_z = &z.front();
			_pk = &pk.front();

			_ei = &ei.front();
			_sw = &sw.front();
			_swz = &swz.front();
			_se = &se.front();
			_swE = &swE.front();
			_ei_cache = &ei_cache.front();

		}

		double * __restrict__ _z;
		double * __restrict__ _pk;

		double * __restrict__ _ei_cache;
		double * __restrict__ _ei;
		double * __restrict__ _sw;
		double * __restrict__ _swz;
		double * __restrict__ _se;
		double * __restrict__ _swE;

	};

	DAClusterizerInZ_vect(const edm::ParameterSet& conf);


	std::vector<std::vector<reco::TransientTrack> >
	clusterize(const std::vector<reco::TransientTrack> & tracks) const;


	std::vector<TransientVertex>
	vertices(const std::vector<reco::TransientTrack> & tracks,
			const int verbosity = 0) const ;

	track_t	fill(const std::vector<reco::TransientTrack> & tracks) const;

	double update(double beta, track_t & gtracks,
			vertex_t & gvertices, bool useRho0, double & rho0) const;

	void dump(const double beta, const vertex_t & y,
			const track_t & tks, const int verbosity = 0) const;
	bool merge(vertex_t &) const;
	bool merge(vertex_t & y, double & beta)const;
	bool purge(vertex_t &, track_t &, double &,
			const double) const;

	void splitAll( vertex_t & y) const;
	bool split(const double beta,  track_t &t, vertex_t & y ) const;

	double beta0(const double betamax, track_t & tks, vertex_t & y) const;

	double Eik( double const& t_z, double const& k_z, double const& t_dz2) const;

	inline double local_exp( double const& inp) const;
	inline void local_exp_list( double* arg_inp, double* arg_out,const int arg_arr_size) const;

private:
	bool verbose_;
	float vertexSize_;
	int maxIterations_;
	double coolingFactor_;
	float betamax_;
	float betastop_;
	double dzCutOff_;
	double d0CutOff_;
	bool use_vdt_;
	bool useTc_;
};


//#ifndef DAClusterizerInZ_new_h
#endif
