#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ_vect.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <cmath>
#include <cassert>
#include <limits>
#include <iomanip>

#include "vdt/vdtMath.h"

using namespace std;

DAClusterizerInZ_vect::DAClusterizerInZ_vect(const edm::ParameterSet& conf) {

	// some defaults to avoid uninitialized variables
	verbose_ = conf.getUntrackedParameter<bool> ("verbose", false);
	use_vdt_ = conf.getUntrackedParameter<bool> ("use_vdt", false);

	betamax_ = 0.1;
	betastop_ = 1.0;
	coolingFactor_ = 0.6;
	maxIterations_ = 100;
	vertexSize_ = 0.01; // 0.1 mm
	dzCutOff_ = 4.0;  

	// configure

	double Tmin = conf.getParameter<double> ("Tmin");
	vertexSize_ = conf.getParameter<double> ("vertexSize");
	coolingFactor_ = conf.getParameter<double> ("coolingFactor");
	useTc_=true;
	if(coolingFactor_<0){
	  coolingFactor_=-coolingFactor_; 
	  useTc_=false;
	}
	d0CutOff_ = conf.getParameter<double> ("d0CutOff");
	dzCutOff_ = conf.getParameter<double> ("dzCutOff");
	maxIterations_ = 100;
	if (Tmin == 0) {
		LogDebug("DAClusterizerinZ_vectorized")  << "DAClusterizerInZ: invalid Tmin" << Tmin
				<< "  reset do default " << 1. / betamax_ << endl;
	} else {
		betamax_ = 1. / Tmin;
	}

}

inline double DAClusterizerInZ_vect::local_exp( double const& inp) const
		{
			if ( use_vdt_)
                          return vdt::fast_exp( inp );

			return std::exp( inp );
		}

inline void DAClusterizerInZ_vect::local_exp_list( double* arg_inp, double* arg_out,const int arg_arr_size) const
		{
			if ( use_vdt_){
//                          std::cout << "I use vdt!\n";
                            for(int i=0; i!=arg_arr_size; ++i ) arg_out[i]=vdt::fast_exp(arg_inp[i]);
		           // vdt::fast_expv(arg_arr_size, arg_inp, arg_out);
                        }
                        else
                          for(int i=0; i!=arg_arr_size; ++i ) arg_out[i]=std::exp(arg_inp[i]);
		}

//todo: use r-value possibility of c++11 here
DAClusterizerInZ_vect::track_t DAClusterizerInZ_vect::fill(const vector<
		reco::TransientTrack> & tracks) const {

	// prepare track data for clustering
	track_t tks;
	for (vector<reco::TransientTrack>::const_iterator it = tracks.begin(); it
			!= tracks.end(); it++)
	{
		double t_pi;
		double t_z = ((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
		double phi=((*it).stateAtBeamLine().trackStateAtPCA()).momentum().phi();
		double tantheta = tan(
				((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
		//  get the beam-spot
		reco::BeamSpot beamspot = (it->stateAtBeamLine()).beamSpot();
 		double t_dz2 = 
		      pow((*it).track().dzError(), 2) // track errror
 		  + (pow(beamspot.BeamWidthX()*cos(phi),2)+pow(beamspot.BeamWidthY()*sin(phi),2))/pow(tantheta,2)  // beam-width
 		  + pow(vertexSize_, 2); // intrinsic vertex size, safer for outliers and short lived decays
		if (d0CutOff_ > 0) {
			Measurement1D IP =
					(*it).stateAtBeamLine().transverseImpactParameter();// error constains beamspot
			t_pi = 1. / (1. + local_exp(pow(IP.value() / IP.error(), 2) - pow(
					d0CutOff_, 2))); // reduce weight for high ip tracks
		} else {
			t_pi = 1.;
		}

		tks.AddItem(t_z, t_dz2, &(*it), t_pi);
	}
        tks.ExtractRaw();

	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized") << "Track count " << tks.GetSize() << std::endl;
	}

	return tks;
}


double DAClusterizerInZ_vect::Eik(double const& t_z, double const& k_z, double const& t_dz2) const
{
	return pow(t_z - k_z, 2) / t_dz2;
}


double DAClusterizerInZ_vect::update(double beta, track_t & gtracks,
		vertex_t & gvertices, bool useRho0, double & rho0) const {

	//update weights and vertex positions
	// mass constrained annealing without noise
	// returns the squared sum of changes of vertex positions

	const unsigned int nt = gtracks.GetSize();
	const unsigned int nv = gvertices.GetSize();

	//initialize sums
	double sumpi = 0;

	// to return how much the prototype moved
	double delta = 0;

	unsigned int itrack;
	unsigned int ivertex;

	// intial value of a sum
	double Z_init = 0;

	// define kernels
	auto kernel_calc_exp_arg = [ &beta, nv ] (
			const unsigned int itrack,
			track_t const& tracks,
			vertex_t const& vertices )
	{
		const double track_z = tracks._z[itrack];
		const double track_dz2 = tracks._dz2[itrack];

		// auto-vectorized
		for ( unsigned int ivertex = 0; ivertex < nv; ++ivertex)
		{
			double mult_res = ( track_z - vertices._z[ivertex] );
			vertices._ei_cache[ivertex] = -beta *
			( mult_res * mult_res ) / track_dz2;
		}
	};

	// auto kernel_add_Z = [ nv, &Z_init ] (vertex_t const& vertices) raises a warning
	// we declare the return type with " -> double"
	auto kernel_add_Z = [ nv, &Z_init ] (vertex_t const& vertices) -> double
	{
		double ZTemp = Z_init;
 
		for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {

			ZTemp += vertices._pk[ivertex] * vertices._ei[ivertex];
		}
		return ZTemp;
	};

	auto kernel_calc_normalization = [ &beta, nv ] (const unsigned int track_num,
			track_t & tks_vec,
			vertex_t & y_vec )
	{
		double w;

		const double tmp_trk_pi = tks_vec._pi[track_num];
		const double tmp_trk_Z_sum = tks_vec._Z_sum[track_num];
		const double tmp_trk_dz2 = tks_vec._dz2[track_num];
		const double tmp_trk_z = tks_vec._z[track_num];

		// auto-vectorized
		for (unsigned int k = 0; k < nv; ++k) {
			y_vec._se[k] += tmp_trk_pi* y_vec._ei[k] / tmp_trk_Z_sum;
			w = y_vec._pk[k] * tmp_trk_pi * y_vec._ei[k] / tmp_trk_Z_sum / tmp_trk_dz2;
			y_vec._sw[k]  += w;
			y_vec._swz[k] += w * tmp_trk_z;
			y_vec._swE[k] += w * y_vec._ei_cache[k]/(-beta);
		}
	};

	// not vectorized
	for (ivertex = 0; ivertex < nv; ++ivertex) {
		gvertices._se[ivertex] = 0.0;
		gvertices._sw[ivertex] = 0.0;
		gvertices._swz[ivertex] = 0.0;
		gvertices._swE[ivertex] = 0.0;
	}


	// independpent of loop
	if ( useRho0 )
	{
		Z_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_); // cut-off
	}

	// loop over tracks
	for (itrack = 0; itrack < nt; ++itrack) {
		kernel_calc_exp_arg(itrack, gtracks, gvertices);
		local_exp_list(gvertices._ei_cache, gvertices._ei, nv);
		//vdt::cephes_single_exp_vect( y_vec._ei, nv );

		gtracks._Z_sum[itrack] = kernel_add_Z(gvertices);

		// used in the next major loop to follow
		if (!useRho0)
			sumpi += gtracks._pi[itrack];

		if (gtracks._Z_sum[itrack] > 0) {
			kernel_calc_normalization(itrack, gtracks, gvertices);
		}
	}

	// now update z and pk
	auto kernel_calc_z = [ &delta, &sumpi, nv, this, useRho0 ] (vertex_t & vertices )
	{
		// does not vectorizes
		for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex )
		{
			if (vertices._sw[ivertex] > 0)
			{
				double znew = vertices._swz[ ivertex ] / vertices._sw[ ivertex ];

				// prevents from vectorizing
				delta += pow( vertices._z[ ivertex ] - znew, 2 );
				vertices._z[ ivertex ] = znew;
			}
			else {
				edm::LogInfo("sumw") << "invalid sum of weights in fit: " << vertices._sw[ivertex]
				<< endl;
				if (this->verbose_) {
					LogDebug("DAClusterizerinZ_vectorized")  << " a cluster melted away ?  pk=" << vertices._pk[ ivertex ] << " sumw="
					<< vertices._sw[ivertex] << endl;
				}
			}

			// dont do, if rho cut
			if ( ! useRho0 )
			{
				vertices._pk[ ivertex ] = vertices._pk[ ivertex ] * vertices._se[ ivertex ] / sumpi;
			}
		}
	};

	kernel_calc_z(gvertices);

	// return how much the prototypes moved
	return delta;
}



bool DAClusterizerInZ_vect::merge(vertex_t & y, double & beta)const{
  // merge clusters that collapsed or never separated,
  // only merge if the estimated critical temperature of the merged vertex is below the current temperature
  // return true if vertices were merged, false otherwise
  const unsigned int nv = y.GetSize();

  if (nv < 2)
    return false;


  for (unsigned int k = 0; (k + 1) < nv; k++) {
    if (fabs(y._z[k + 1] - y._z[k]) < 2.e-2) {
      double rho=y._pk[k] + y._pk[k+1];
      double swE=y._swE[k]+y._swE[k+1] - y._pk[k]*y._pk[k+1] /rho *pow(y._z[k+1]-y._z[k],2);
      double Tc=2*swE/(y._sw[k]+y._sw[k+1]);

      if(Tc*beta<1){
        if(rho>0){
	  y._z[k] = (y._pk[k]*y._z[k] + y._pk[k+1]*y._z[k + 1])/rho;
        }else{
	  y._z[k] = 0.5 * (y._z[k] + y._z[k + 1]);
        }
        y._pk[k] = rho;
        y._sw[k]+=y._sw[k+1];
        y._swE[k]=swE;
        y.RemoveItem(k+1);
        return true;
      }
    }
  }

  return false;
}





bool DAClusterizerInZ_vect::merge(vertex_t & y) const {
	// merge clusters that collapsed or never separated, return true if vertices were merged, false otherwise

	const unsigned int nv = y.GetSize();

	if (nv < 2)
		return false;

	for (unsigned int k = 0; (k + 1) < nv; k++) {
		//if ((k+1)->z - k->z<1.e-2){  // note, no fabs here, maintains z-ordering  (with split()+merge() at every temperature)
		if (fabs(y._z[k + 1] - y._z[k]) < 1.e-2) { // with fabs if only called after freeze-out (splitAll() at highter T)
			y._pk[k] += y._pk[k + 1];
			y._z[k] = 0.5 * (y._z[k] + y._z[k + 1]);

			y.RemoveItem(k + 1);

			if ( verbose_)
			{
				LogDebug("DAClusterizerinZ_vectorized") << "Merging vertices k = " << k << std::endl;
			}

			return true;
		}
	}

	return false;
}

bool DAClusterizerInZ_vect::purge(vertex_t & y, track_t & tks, double & rho0,
		const double beta) const {
	// eliminate clusters with only one significant/unique track
	const unsigned int nv = y.GetSize();
	const unsigned int nt = tks.GetSize();

	if (nv < 2)
		return false;

	double sumpmin = nt;
	unsigned int k0 = nv;

	for (unsigned int k = 0; k < nv; k++) {

		int nUnique = 0;
		double sump = 0;
		double pmax = y._pk[k] / (y._pk[k] + rho0 * local_exp(-beta * dzCutOff_
				* dzCutOff_));

		for (unsigned int i = 0; i < nt; i++) {

			if (tks._Z_sum[i] > 0) {
				//double p=pik(beta,tks[i],*k);
				double p = y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],
						tks._dz2[i])) / tks._Z_sum[i];
				sump += p;
				if ((p > 0.9 * pmax) && (tks._pi[i] > 0)) {
					nUnique++;
				}
			}
		}

		if ((nUnique < 2) && (sump < sumpmin)) {
			sumpmin = sump;
			k0 = k;
		}
	}

	if (k0 != nv) {
		if (verbose_) {
			LogDebug("DAClusterizerinZ_vectorized")  << "eliminating prototype at " << y._z[k0] << " with sump="
					<< sumpmin << endl;
		}
		//rho0+=k0->pk;
		y.RemoveItem(k0);
		return true;
	} else {
		return false;
	}
}

double DAClusterizerInZ_vect::beta0(double betamax, track_t & tks, vertex_t & y) const {

	double T0 = 0; // max Tc for beta=0
	// estimate critical temperature from beta=0 (T=inf)
	const unsigned int nt = tks.GetSize();
	const unsigned int nv = y.GetSize();

	for (unsigned int k = 0; k < nv; k++) {

		// vertex fit at T=inf
		double sumwz = 0;
		double sumw = 0;
		for (unsigned int i = 0; i < nt; i++) {
			double w = tks._pi[i] / tks._dz2[i];
			sumwz += w * tks._z[i];
			sumw += w;
		}
		y._z[k] = sumwz / sumw;

		// estimate Tcrit, eventually do this in the same loop
		double a = 0, b = 0;
		for (unsigned int i = 0; i < nt; i++) {
			double dx = tks._z[i] - (y._z[k]);
			double w = tks._pi[i] / tks._dz2[i];
			a += w * pow(dx, 2) / tks._dz2[i];
			b += w;
		}
		double Tc = 2. * a / b; // the critical temperature of this vertex
		if (Tc > T0)
			T0 = Tc;
	}// vertex loop (normally there should be only one vertex at beta=0)

	if (T0 > 1. / betamax) {
		return betamax / pow(coolingFactor_, int(log(T0 * betamax) / log(
				coolingFactor_)) - 1);
	} else {
		// ensure at least one annealing step
		return betamax / coolingFactor_;
	}
}


bool DAClusterizerInZ_vect::split(const double beta,  track_t &tks, vertex_t & y ) const{
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split

  double epsilon=1e-3;      // split all single vertices by 10 um
  unsigned int nv = y.GetSize();

  // avoid left-right biases by splitting highest Tc first

  std::vector<std::pair<double, unsigned int> > critical;
  for(unsigned int k=0; k<nv; k++){
    double Tc= 2*y._swE[k]/y._sw[k];
    if (beta*Tc > 1.){
      critical.push_back( make_pair(Tc, k));
    }
  }
  if (critical.size()==0) return false;
  stable_sort(critical.begin(), critical.end(), std::greater<std::pair<double, unsigned int> >() );


  bool split=false;
  const unsigned int nt = tks.GetSize();

  for(unsigned int ic=0; ic<critical.size(); ic++){
    unsigned int k=critical[ic].second;
    // estimate subcluster positions and weight
    double p1=0, z1=0, w1=0;
    double p2=0, z2=0, w2=0;
    for(unsigned int i=0; i<nt; i++){
      if (tks._Z_sum[i] > 0) {
	double p = y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],
						    tks._dz2[i])) / tks._Z_sum[i];

        double w=p/tks._dz2[i];
        if(tks._z[i]<y._z[k]){
          p1+=p; z1+=w*tks._z[i]; w1+=w;
        }else{
          p2+=p; z2+=w*tks._z[i]; w2+=w;
        }
      }
    }
    if(w1>0){  z1=z1/w1;} else{z1=y._z[k]-epsilon;}
    if(w2>0){  z2=z2/w2;} else{z2=y._z[k]+epsilon;}

    // reduce split size if there is not enough room
    if( ( k   > 0 ) && ( y._z[k-1]>=z1 ) ){ z1=0.5*(y._z[k]+y._z[k-1]); }
    if( ( k+1 < nv) && ( y._z[k+1]<=z2 ) ){ z2=0.5*(y._z[k]+y._z[k+1]); }

    // split if the new subclusters are significantly separated
    if( (z2-z1)>epsilon){
      split=true;
      double pk1=p1*y._pk[k]/(p1+p2);
      double pk2=p2*y._pk[k]/(p1+p2);
      y._z[k]  =  z2;
      y._pk[k] = pk2;
      y.InsertItem(k, z1, pk1);
      nv++;

     // adjust remaining pointers
      for(unsigned int jc=ic; jc<critical.size(); jc++){
        if (critical[jc].second>k) {critical[jc].second++;}
      }
    }
  }
  return split;
}



void DAClusterizerInZ_vect::splitAll( vertex_t & y) const {

	const unsigned int nv = y.GetSize();

	double epsilon = 1e-3; // split all single vertices by 10 um
	double zsep = 2 * epsilon; // split vertices that are isolated by at least zsep (vertices that haven't collapsed)
	vertex_t y1;

	if (verbose_)
	{
		LogDebug("DAClusterizerinZ_vectorized") << "Before Split "<< std::endl;
		y.DebugOut();
	}

	for (unsigned int k = 0; k < nv; k++) {

		if (
				( (k == 0)       	|| ( y._z[k - 1]	< (y._z[k] - zsep)) ) &&
		        ( ((k + 1) == nv)	|| ( y._z[k + 1] 	> (y._z[k] + zsep)) )   )
		{
			// isolated prototype, split
			double new_z = y.z[k] - epsilon;
			y.z[k] = y.z[k] + epsilon;

			double new_pk = 0.5 * y.pk[k];
			y.pk[k] = 0.5 * y.pk[k];

			y1.AddItem(new_z, new_pk);
			y1.AddItem(y._z[k], y._pk[k]);

		}
		else if ( (y1.GetSize() == 0 ) ||
				(y1._z[y1.GetSize() - 1] <  (y._z[k] - zsep)  ))
		{

			y1.AddItem(y._z[k], y._pk[k]);
		}
		else
		{
			y1._z[y1.GetSize() - 1] = y1._z[y1.GetSize() - 1] - epsilon;
			y._z[k] = y._z[k] + epsilon;
			y1.AddItem( y._z[k] , y._pk[k]);
		}
	}// vertex loop

	y = y1;
	y.ExtractRaw();

	if (verbose_)
	{
		LogDebug("DAClusterizerinZ_vectorized") << "After split " << std::endl;
		y.DebugOut();
	}
}


void DAClusterizerInZ_vect::dump(const double beta, const vertex_t & y,
		const track_t & tks, int verbosity) const {

	const unsigned int nv = y.GetSize();
	const unsigned int nt = tks.GetSize();

	LogDebug("DAClusterizerinZ_vectorized")  << "-----DAClusterizerInZ::dump ----" << endl;
	LogDebug("DAClusterizerinZ_vectorized")  << "beta=" << beta << "   betamax= " << betamax_ << endl;
	LogDebug("DAClusterizerinZ_vectorized")  << "                                                               z= ";
	LogDebug("DAClusterizerinZ_vectorized")  << setprecision(4);
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		LogDebug("DAClusterizerinZ_vectorized")  << setw(8) << fixed << y._z[ivertex];
	}
	LogDebug("DAClusterizerinZ_vectorized")  << endl << "T=" << setw(15) << 1. / beta
			<< "                                            Tc= ";
	LogDebug("DAClusterizerinZ_vectorized")  << endl
			<< "                                                               pk=";
	double sumpk = 0;
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		LogDebug("DAClusterizerinZ_vectorized")  << setw(8) << setprecision(3) << fixed << y._pk[ivertex];
		sumpk += y._pk[ivertex];
	}
	LogDebug("DAClusterizerinZ_vectorized")  << endl;

	if (verbosity > 0) {
		double E = 0, F = 0;
		LogDebug("DAClusterizerinZ_vectorized")  << endl;
		LogDebug("DAClusterizerinZ_vectorized") 
		<< "----       z +/- dz                ip +/-dip       pt    phi  eta    weights  ----"
		<< endl;
		LogDebug("DAClusterizerinZ_vectorized")  << setprecision(4);
		for (unsigned int i = 0; i < nt; i++) {
			if (tks._Z_sum[i] > 0) {
				F -= log(tks._Z_sum[i]) / beta;
			}
			double tz = tks._z[i];
			LogDebug("DAClusterizerinZ_vectorized")  << setw(3) << i << ")" << setw(8) << fixed << setprecision(4)
			 << tz << " +/-" << setw(6) << sqrt(tks._dz2[i]);

			if (tks.tt[i]->track().quality(reco::TrackBase::highPurity)) {
				LogDebug("DAClusterizerinZ_vectorized")  << " *";
			} else {
				LogDebug("DAClusterizerinZ_vectorized")  << "  ";
			}
			if (tks.tt[i]->track().hitPattern().hasValidHitInFirstPixelBarrel()) {
				LogDebug("DAClusterizerinZ_vectorized")  << "+";
			} else {
				LogDebug("DAClusterizerinZ_vectorized")  << "-";
			}
			LogDebug("DAClusterizerinZ_vectorized")  << setw(1)
			 << tks.tt[i]->track().hitPattern().pixelBarrelLayersWithMeasurement(); // see DataFormats/TrackReco/interface/HitPattern.h
			LogDebug("DAClusterizerinZ_vectorized")  << setw(1)
			 << tks.tt[i]->track().hitPattern().pixelEndcapLayersWithMeasurement();
			LogDebug("DAClusterizerinZ_vectorized")  << setw(1) << hex
					<< tks.tt[i]->track().hitPattern().trackerLayersWithMeasurement()
					- tks.tt[i]->track().hitPattern().pixelLayersWithMeasurement()
					<< dec;
			LogDebug("DAClusterizerinZ_vectorized")  << "=" << setw(1) << hex
					<< tks.tt[i]->track().trackerExpectedHitsOuter().numberOfHits()
					<< dec;

			Measurement1D IP =
					tks.tt[i]->stateAtBeamLine().transverseImpactParameter();
			LogDebug("DAClusterizerinZ_vectorized")  << setw(8) << IP.value() << "+/-" << setw(6) << IP.error();
			LogDebug("DAClusterizerinZ_vectorized")  << " " << setw(6) << setprecision(2)
			 << tks.tt[i]->track().pt() * tks.tt[i]->track().charge();
			LogDebug("DAClusterizerinZ_vectorized")  << " " << setw(5) << setprecision(2)
			 << tks.tt[i]->track().phi() << " " << setw(5)
			 << setprecision(2) << tks.tt[i]->track().eta();

			double sump = 0.;
			for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
				if ((tks._pi[i] > 0) && (tks._Z_sum[i] > 0)) {
					//double p=pik(beta,tks[i],*k);
					double p = y._pk[ivertex]  * exp(-beta * Eik(tks._z[i], y._z[ivertex], tks._dz2[i])) / tks._Z_sum[i];
					if (p > 0.0001) {
						LogDebug("DAClusterizerinZ_vectorized")  << setw(8) << setprecision(3) << p;
					} else {
						LogDebug("DAClusterizerinZ_vectorized")  << "    .   ";
					}
					E += p * Eik(tks._z[i], y._z[ivertex], tks._dz2[i]);
					sump += p;
				} else {
					LogDebug("DAClusterizerinZ_vectorized")  << "        ";
				}
			}
			LogDebug("DAClusterizerinZ_vectorized")  << endl;
		}
		LogDebug("DAClusterizerinZ_vectorized")  << endl << "T=" << 1 / beta << " E=" << E << " n=" << y.GetSize()
			 << "  F= " << F << endl << "----------" << endl;
	}
}

vector<TransientVertex> DAClusterizerInZ_vect::vertices(const vector<
		reco::TransientTrack> & tracks, const int verbosity) const {


	track_t tks = fill(tracks);
	tks.ExtractRaw();

	unsigned int nt = tracks.size();
	double rho0 = 0.0; // start with no outlier rejection

	vector<TransientVertex> clusters;
	if (tks.GetSize() == 0)
		return clusters;

	vertex_t y; // the vertex prototypes

	// initialize:single vertex at infinite temperature
	y.AddItem( 0, 1.0);

	int niter = 0; // number of iterations


	// estimate first critical temperature
	double beta = beta0(betamax_, tks, y);
	if ( verbose_)
	{
		LogDebug("DAClusterizerinZ_vectorized") << "Beta0 is " << beta << std::endl;
	}

	niter = 0;
	while ((update(beta, tks, y, false, rho0) > 1.e-6) && (niter++
			< maxIterations_)) {
	}

	// annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)
	while (beta < betamax_) {


		if(useTc_){
		  update(beta, tks,y, false, rho0);
		  while(merge(y,beta)){update(beta, tks,y, false, rho0);}
		  split(beta, tks,y);
		  beta=beta/coolingFactor_;
		}else{
		  beta=beta/coolingFactor_;
		  splitAll(y);
		}

		// make sure we are not too far from equilibrium before cooling further
		niter = 0;
		while ((update(beta, tks, y, false, rho0) > 1.e-6) && (niter++
				< maxIterations_)) {
		}

	}


	if(useTc_){
	  // last round of splitting, make sure no critical clusters are left
	  update(beta, tks,y, false, rho0);// make sure Tc is up-to-date
	  while(merge(y,beta)){update(beta, tks,y, false, rho0);}
	  unsigned int ntry=0;
	  while( split(beta, tks,y) && (ntry++<10) ){
	    niter=0; 
	    while((update(beta, tks,y, false, rho0)>1.e-6)  && (niter++ < maxIterations_)){}
	    merge(y,beta);
	    update(beta, tks,y, false, rho0);
	  }
	}else{
	  // merge collapsed clusters 
	  while(merge(y,beta)){update(beta, tks,y, false, rho0);}  
	  //while(merge(y)){}   original code
	}
 
	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized")  << "dump after 1st merging " << endl;
		dump(beta, y, tks, 2);
	}

	// switch on outlier rejection
	rho0 = 1. / nt;
	
	// auto-vectorized
	for (unsigned int k = 0; k < y.GetSize(); k++) {
		y._pk[k] = 1.;
	} // democratic
	niter = 0;
	while ((update(beta, tks, y, true, rho0) > 1.e-8) && (niter++
			< maxIterations_)) {
	}
	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized")  << "rho0=" << rho0 << " niter=" << niter << endl;
		dump(beta, y, tks, 2);
	}

	// merge again  (some cluster split by outliers collapse here)
	while (merge(y)) {
	}
	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized")  << "dump after 2nd merging " << endl;
		dump(beta, y, tks, 2);
	}

	// continue from freeze-out to Tstop (=1) without splitting, eliminate insignificant vertices
	while (beta <= betastop_) {
		while (purge(y, tks, rho0, beta)) {
			niter = 0;
			while ((update(beta, tks, y, true, rho0) > 1.e-6) && (niter++
					< maxIterations_)) {
			}
		}
		beta /= coolingFactor_;
		niter = 0;
		while ((update(beta, tks, y, true, rho0) > 1.e-6) && (niter++
				< maxIterations_)) {
		}
	}

	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized")  << "Final result, rho0=" << rho0 << endl;
		dump(beta, y, tks, 2);
	}

	// select significant tracks and use a TransientVertex as a container
	GlobalError dummyError;

	// ensure correct normalization of probabilities, should makes double assginment reasonably impossible
	const unsigned int nv = y.GetSize();

	for (unsigned int i = 0; i < nt; i++) {
		tks._Z_sum[i] = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);

		for (unsigned int k = 0; k < nv; k++) {
			tks._Z_sum[i] += y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],
					tks._dz2[i]));
		}
	}

	for (unsigned int k = 0; k < nv; k++) {

		GlobalPoint pos(0, 0, y._z[k]);

		vector<reco::TransientTrack> vertexTracks;
		for (unsigned int i = 0; i < nt; i++) {
			if (tks._Z_sum[i] > 0) {

				double p = y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],
						tks._dz2[i])) / tks._Z_sum[i];
				if ((tks._pi[i] > 0) && (p > 0.5)) {

					vertexTracks.push_back(*(tks.tt[i]));
					tks._Z_sum[i] = 0;
				} // setting Z=0 excludes double assignment
			}
		}
		TransientVertex v(pos, dummyError, vertexTracks, 0);
		clusters.push_back(v);
	}

	return clusters;

}

vector<vector<reco::TransientTrack> > DAClusterizerInZ_vect::clusterize(
		const vector<reco::TransientTrack> & tracks) const {
                  
	if (verbose_) {
		std::cout  << "###################################################" << endl;
		std::cout  << "# vectorized DAClusterizerInZ_vect::clusterize   nt=" << tracks.size() << endl;
		std::cout  << "###################################################" << endl;
	}

	vector<vector<reco::TransientTrack> > clusters;
	vector<TransientVertex> pv = vertices(tracks);

	if (verbose_) {
		LogDebug("DAClusterizerinZ_vectorized")  << "# DAClusterizerInZ::clusterize   pv.size=" << pv.size()
				<< endl;
	}
	if (pv.size() == 0) {
		return clusters;
	}

	// fill into clusters and merge
	vector<reco::TransientTrack> aCluster = pv.begin()->originalTracks();

	for (vector<TransientVertex>::iterator k = pv.begin() + 1; k != pv.end(); k++) {
	        if ( fabs(k->position().z() - (k - 1)->position().z()) > (2 * vertexSize_)) {
			// close a cluster
			clusters.push_back(aCluster);
			aCluster.clear();
		}
		for (unsigned int i = 0; i < k->originalTracks().size(); i++) {
			aCluster.push_back(k->originalTracks().at(i));
		}

	}
	clusters.push_back(aCluster);

	return clusters;

}

