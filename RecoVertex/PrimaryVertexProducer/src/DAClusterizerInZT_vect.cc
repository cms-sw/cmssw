#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZT_vect.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <cmath>
#include <cassert>
#include <limits>
#include <iomanip>
#include "FWCore/Utilities/interface/isFinite.h"
#include "vdt/vdtMath.h"

using namespace std;

DAClusterizerInZT_vect::DAClusterizerInZT_vect(const edm::ParameterSet& conf) {

  // hardcoded parameters
  maxIterations_ = 100;
  mintrkweight_ = 0.5; // conf.getParameter<double>("mintrkweight");


  // configurable debug outptut debug output
  verbose_ = conf.getUntrackedParameter<bool> ("verbose", false);
  zdumpcenter_ = conf.getUntrackedParameter<double> ("zdumpcenter", 0.);
  zdumpwidth_ = conf.getUntrackedParameter<double> ("zdumpwidth", 20.);
  
  // configurable parameters
  double minT = conf.getParameter<double> ("Tmin")*std::sqrt(2.0);
  double purgeT = conf.getParameter<double> ("Tpurge")*std::sqrt(2.0);
  double stopT = conf.getParameter<double> ("Tstop")*std::sqrt(2.0);
  vertexSize_ = conf.getParameter<double> ("vertexSize");
  vertexSizeTime_ = conf.getParameter<double> ("vertexSizeTime");
  coolingFactor_ = conf.getParameter<double> ("coolingFactor");
  useTc_=true;
  if(coolingFactor_<0){
    coolingFactor_=-coolingFactor_; 
    useTc_=false;
  }
  coolingFactor_ = std::sqrt(coolingFactor_);
  d0CutOff_ = conf.getParameter<double> ("d0CutOff");
  dzCutOff_ = conf.getParameter<double> ("dzCutOff");
  dtCutOff_ = conf.getParameter<double> ("dtCutOff");
  uniquetrkweight_ = conf.getParameter<double>("uniquetrkweight");
  zmerge_ = conf.getParameter<double>("zmerge");
  tmerge_ = conf.getParameter<double>("tmerge");

#ifdef VI_DEBUG
  if(verbose_){
    std::cout << "DAClusterizerinZT_vect: mintrkweight = " << mintrkweight_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: uniquetrkweight = " << uniquetrkweight_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: zmerge = " << zmerge_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: tmerge = " << tmerge_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: Tmin = " << minT << std::endl;
    std::cout << "DAClusterizerinZT_vect: Tpurge = " << purgeT << std::endl;
    std::cout << "DAClusterizerinZT_vect: Tstop = " << stopT << std::endl;
    std::cout << "DAClusterizerinZT_vect: vertexSize = " << vertexSize_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: vertexSizeTime = " << vertexSizeTime_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: coolingFactor = " << coolingFactor_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: d0CutOff = " << d0CutOff_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: dzCutOff = " << dzCutOff_ << std::endl;
    std::cout << "DAClusterizerinZT_vect: dtCutoff = " << dtCutOff_ << std::endl;
  }
#endif


  if (minT == 0) {
    edm::LogWarning("DAClusterizerinZT_vectorized") << "DAClusterizerInZT: invalid Tmin" << minT
						   << "  reset do default " << 1. / betamax_;
  } else {
    betamax_ = 1. / minT;
  }


  if ((purgeT > minT) || (purgeT == 0)) {
    edm::LogWarning("DAClusterizerinZT_vectorized") << "DAClusterizerInZT: invalid Tpurge" << purgeT
						   << "  set to " << minT;
    purgeT =  minT;
  }
  betapurge_ = 1./purgeT;
  

  if ((stopT > purgeT) || (stopT == 0)) {
    edm::LogWarning("DAClusterizerinZT_vectorized") << "DAClusterizerInZT: invalid Tstop" << stopT
						   << "  set to  " << max(1., purgeT);
    stopT = max(1., purgeT) ;
  }
  betastop_ = 1./stopT;
  
}


namespace {
  inline double local_exp( double const& inp) {
    return vdt::fast_exp( inp );
  }
  
  inline void local_exp_list( double const *arg_inp, 
			      double *arg_out, 
			      const unsigned arg_arr_size) {
    for(unsigned i=0; i!=arg_arr_size; ++i ) arg_out[i]=vdt::fast_exp(arg_inp[i]);
  }

}

//todo: use r-value possibility of c++11 here
DAClusterizerInZT_vect::track_t 
DAClusterizerInZT_vect::fill(const vector<reco::TransientTrack> & tracks) const {

  // prepare track data for clustering
  track_t tks;
  for( const auto& tk : tracks ) {
    if (!tk.isValid()) continue;
    double t_pi=1.;
    double t_z = tk.stateAtBeamLine().trackStateAtPCA().position().z();
    double t_t = tk.timeExt();
    if (std::fabs(t_z) > 1000.) continue;
    auto const & t_mom = tk.stateAtBeamLine().trackStateAtPCA().momentum();
    //  get the beam-spot
    reco::BeamSpot beamspot = tk.stateAtBeamLine().beamSpot();
    double t_dz2 = 
      std::pow(tk.track().dzError(), 2) // track errror
      + (std::pow(beamspot.BeamWidthX()*t_mom.x(),2)+std::pow(beamspot.BeamWidthY()*t_mom.y(),2))*std::pow(t_mom.z(),2)/std::pow(t_mom.perp2(),2) // beam spot width
      + std::pow(vertexSize_, 2); // intrinsic vertex size, safer for outliers and short lived decays
    t_dz2 = 1./ t_dz2;
    double t_dt2 =std::pow(tk.dtErrorExt(),2.) + std::pow(vertexSizeTime_,2.); // the ~injected~ timing error, need to add a small minimum vertex size in time
    t_dt2 = 1./t_dt2;
    if (edm::isNotFinite(t_dz2) || t_dz2 < std::numeric_limits<double>::min() ) continue;
    if (edm::isNotFinite(t_dt2) || t_dt2 < std::numeric_limits<double>::min() ) continue;
    if (d0CutOff_ > 0) {
      Measurement1D atIP =
	tk.stateAtBeamLine().transverseImpactParameter();// error contains beamspot
      t_pi = 1. / (1. + local_exp(std::pow(atIP.value() / atIP.error(), 2) - std::pow(d0CutOff_, 2))); // reduce weight for high ip tracks
      if (edm::isNotFinite(t_pi) ||  t_pi < std::numeric_limits<double>::epsilon())  continue; // usually is > 0.99
    }
    LogTrace("DAClusterizerinZT_vectorized") << t_z << ' ' << t_t <<' '<< t_dz2 << ' ' << t_dt2 <<' '<< t_pi;
    tks.addItem(t_z, t_t, t_dz2, t_dt2, &tk, t_pi);
  }
  tks.extractRaw();
  
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout << "Track count " << tks.getSize() << std::endl;
  }
#endif
  
  return tks;
}


namespace {
  inline
  double Eik(double t_z, double k_z, double t_dz2, double t_t, double k_t, double t_dt2) {
    return std::pow(t_z - k_z, 2) * t_dz2 + std::pow(t_t - k_t,2) * t_dt2;
  }
}

double DAClusterizerInZT_vect::update(double beta, track_t & gtracks,
				     vertex_t & gvertices, bool useRho0, const double & rho0) const {

  //update weights and vertex positions
  // mass constrained annealing without noise
  // returns the squared sum of changes of vertex positions
  
  const unsigned int nt = gtracks.getSize();
  const unsigned int nv = gvertices.getSize();
  
  //initialize sums
  double sumpi = 0.;
  
  // to return how much the prototype moved
  double delta = 0.;
  

  // intial value of a sum
  double Z_init = 0;
  // independpent of loop
  if ( useRho0 )
    {
      Z_init = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_); // cut-off
    }
  
  // define kernels
  auto kernel_calc_exp_arg = [ beta, nv ] ( const unsigned int itrack,
					     track_t const& tracks,
					     vertex_t const& vertices ) {
    
    const auto track_z = tracks.z_[itrack];
    const auto track_t = tracks.t_[itrack];
    const auto botrack_dz2 = -beta*tracks.dz2_[itrack];
    const auto botrack_dt2 = -beta*tracks.dt2_[itrack];

    // auto-vectorized
    for ( unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      const auto mult_resz = track_z - vertices.z_[ivertex];
      const auto mult_rest = track_t - vertices.t_[ivertex];
      vertices.ei_cache_[ivertex] = botrack_dz2 * ( mult_resz * mult_resz ) + botrack_dt2 * ( mult_rest * mult_rest );
    }
  };
  
  auto kernel_add_Z = [ nv, Z_init ] (vertex_t const& vertices) -> double
    {
      double ZTemp = Z_init;
      for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {	
	ZTemp += vertices.pk_[ivertex] * vertices.ei_[ivertex];
      }
      return ZTemp;
    };

  auto kernel_calc_normalization = [ beta, nv ] (const unsigned int track_num,
						  track_t & tks_vec,
						  vertex_t & y_vec ) {
    auto tmp_trk_pi = tks_vec.pi_[track_num];
    auto o_trk_Z_sum = 1./tks_vec.Z_sum_[track_num];
    auto o_trk_err_sum = tks_vec.errsum_[track_num];
    auto tmp_trk_z = tks_vec.z_[track_num];
    auto tmp_trk_t = tks_vec.t_[track_num];
    auto obeta =  -1./beta;
    
    // auto-vectorized
    for (unsigned int k = 0; k < nv; ++k) {
      // parens are important for numerical stability
      y_vec.se_[k] +=  tmp_trk_pi*( y_vec.ei_[k] * o_trk_Z_sum );      
      const auto w = tmp_trk_pi * (y_vec.pk_[k] * y_vec.ei_[k] * o_trk_Z_sum) * o_trk_err_sum; 
      y_vec.sw_[k]  += w;
      y_vec.swz_[k] += w * tmp_trk_z;
      y_vec.swt_[k] += w * tmp_trk_t;
      y_vec.swE_[k] += w * y_vec.ei_cache_[k]*obeta;
    }
  };
  
  
  for (auto ivertex = 0U; ivertex < nv; ++ivertex) {
    gvertices.se_[ivertex] = 0.0;
    gvertices.sw_[ivertex] = 0.0;
    gvertices.swz_[ivertex] = 0.0;
    gvertices.swt_[ivertex] = 0.0;
    gvertices.swE_[ivertex] = 0.0;
  }
  
   
  // loop over tracks
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    kernel_calc_exp_arg(itrack, gtracks, gvertices);
    local_exp_list(gvertices.ei_cache_, gvertices.ei_, nv);
        
    gtracks.Z_sum_[itrack] = kernel_add_Z(gvertices);
    if (edm::isNotFinite(gtracks.Z_sum_[itrack])) gtracks.Z_sum_[itrack] = 0.0;
    // used in the next major loop to follow
    sumpi += gtracks.pi_[itrack];
    
    if (gtracks.Z_sum_[itrack] > 1.e-100){
      kernel_calc_normalization(itrack, gtracks, gvertices);
    }
  }
  
  // now update z, t, and pk
  auto kernel_calc_zt = [  sumpi, nv, this, useRho0 ] (vertex_t & vertices ) -> double {
    
    double delta=0;
    // does not vectorizes
    for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex ) {
      if (vertices.sw_[ivertex] > 0.) {
	auto znew = vertices.swz_[ ivertex ] / vertices.sw_[ ivertex ];
	// prevents from vectorizing if 
	delta += std::pow( vertices.z_[ ivertex ] - znew, 2 );
	vertices.z_[ ivertex ] = znew;
        auto tnew = vertices.swt_[ ivertex ] / vertices.sw_[ ivertex ];
	// prevents from vectorizing if 
	delta += std::pow( vertices.t_[ ivertex ] - tnew, 2 );
        vertices.t_[ ivertex ] = tnew;
      }         
#ifdef VI_DEBUG
      else {
	edm::LogInfo("sumw") << "invalid sum of weights in fit: " << vertices.sw_[ivertex] << endl;
	if (this->verbose_) {
	  std::cout  << " a cluster melted away ?  pk=" << vertices.pk_[ ivertex ] << " sumw="
		     << vertices.sw_[ivertex] << endl;
	}
      }
#endif
    }

    auto osumpi = 1./sumpi;
    for (unsigned int ivertex = 0; ivertex < nv; ++ivertex )
      vertices.pk_[ ivertex ] = vertices.pk_[ ivertex ] * vertices.se_[ ivertex ] * osumpi;

    return delta;
  };
  
  delta += kernel_calc_zt(gvertices);
  
  // return how much the prototypes moved
  return delta;
}





bool DAClusterizerInZT_vect::merge(vertex_t & y, double & beta)const{
  // merge clusters that collapsed or never separated,
  // only merge if the estimated critical temperature of the merged vertex is below the current temperature
  // return true if vertices were merged, false otherwise
  const unsigned int nv = y.getSize();

  if (nv < 2)
    return false;

  // merge the smallest distance clusters first
  std::vector<std::pair<double, unsigned int> > critical;
  for (unsigned int k = 0; (k + 1) < nv; k++) {
    if ( std::fabs(y.z_[k + 1] - y.z_[k]) < zmerge_ &&
         std::fabs(y.t_[k + 1] - y.t_[k]) < tmerge_    ) {
      auto dt2 = std::pow(y.t_[k + 1] - y.t_[k],2);
      auto dz2 = std::pow(y.z_[k + 1] - y.z_[k],2);
      critical.push_back( make_pair( dz2 + dt2, k ) );
    }
  }
  if (critical.empty()) return false;

  std::stable_sort(critical.begin(), critical.end(), std::less<std::pair<double, unsigned int> >() );


  for (unsigned int ik=0; ik < critical.size(); ik++){
    unsigned int k = critical[ik].second;
    double rho = y.pk_[k]+y.pk_[k+1];
    double swE = y.swE_[k]+y.swE_[k+1]-y.pk_[k]*y.pk_[k+1] / rho*( std::pow(y.z_[k+1]-y.z_[k],2) + std::pow(y.t_[k+1]-y.t_[k],2) );
    double Tc = 2*swE / (y.sw_[k]+y.sw_[k+1]);

    if(Tc*beta < 1){
#ifdef VI_DEBUG
      if(verbose_){ std::cout << "merging (" << y.z_[k + 1] << ',' << y.t_[k + 1] << ") and (" <<  y.z_[k] << ',' << y.t_[k] << ")  Tc = " << Tc <<  "  sw = "  << y.sw_[k]+y.sw_[k+1]  <<std::endl;}
#endif
      if(rho > 0){
	y.z_[k] = (y.pk_[k]*y.z_[k] + y.pk_[k+1]*y.z_[k + 1])/rho;
        y.t_[k] = (y.pk_[k]*y.t_[k] + y.pk_[k+1]*y.t_[k + 1])/rho;
      }else{
	y.z_[k] = 0.5 * (y.z_[k] + y.z_[k + 1]);
        y.t_[k] = 0.5 * (y.t_[k] + y.t_[k + 1]);
      }
      y.pk_[k] = rho;
      y.sw_[k] += y.sw_[k+1];
      y.swE_[k] = swE;
      y.removeItem(k+1);
      return true;
    }
  }

  return false;
}




bool 
DAClusterizerInZT_vect::purge(vertex_t & y, track_t & tks, double & rho0, const double beta) const {
  constexpr double eps = 1.e-100;
  // eliminate clusters with only one significant/unique track
  const unsigned int nv = y.getSize();
  const unsigned int nt = tks.getSize();
  
  if (nv < 2)
    return false;
  
  double sumpmin = nt;
  unsigned int k0 = nv;
  
  int nUnique = 0;
  double sump = 0;

  std::vector<double> inverse_zsums(nt), arg_cache(nt), eik_cache(nt);
  double * pinverse_zsums;
  double * parg_cache;
  double * peik_cache;
  pinverse_zsums = inverse_zsums.data();
  parg_cache = arg_cache.data();
  peik_cache = eik_cache.data();
  for(unsigned i = 0; i < nt; ++i) {
    inverse_zsums[i] = tks.Z_sum_[i] > eps ? 1./tks.Z_sum_[i] : 0.0;
  }

  for (unsigned int k = 0; k < nv; ++k) {
    
    nUnique = 0;
    sump = 0;

    const double pmax = y.pk_[k] / (y.pk_[k] + rho0 * local_exp(-beta * dzCutOff_* dzCutOff_));
    const double pcut = uniquetrkweight_ * pmax;
    for(unsigned i = 0; i < nt; ++i) {
      const auto track_z = tks.z_[i];
      const auto track_t = tks.t_[i];
      const auto botrack_dz2 = -beta*tks.dz2_[i];
      const auto botrack_dt2 = -beta*tks.dt2_[i];
      
      const auto mult_resz = track_z - y.z_[k];
      const auto mult_rest = track_t - y.t_[k];
      parg_cache[i] = botrack_dz2 * ( mult_resz * mult_resz ) + botrack_dt2 * ( mult_rest * mult_rest );
    }
    local_exp_list(parg_cache, peik_cache, nt);
    for (unsigned int i = 0; i < nt; ++i) {
      const double p = y.pk_[k] * peik_cache[i] * pinverse_zsums[i];
      sump += p;
      nUnique += ( ( p > pcut ) & ( tks.pi_[i] > 0 ) );
    }

    if ((nUnique < 2) && (sump < sumpmin)) {
      sumpmin = sump;
      k0 = k;
    }

  }
  
  if (k0 != nv) {
#ifdef VI_DEBUG
    if (verbose_) {
      std::cout  << "eliminating prototype at " << std::setw(10) << std::setprecision(4) << y.z_[k0] 
		 << " with sump=" << sumpmin
		 << "  rho*nt =" << y.pk_[k0]*nt
		 << endl;
    }
#endif
    y.removeItem(k0);
    return true;
  } else {
    return false;
  }
}




double 
DAClusterizerInZT_vect::beta0(double betamax, track_t const  & tks, vertex_t const & y) const {
  
  double T0 = 0; // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  const unsigned int nt = tks.getSize();
  const unsigned int nv = y.getSize();
  
  for (unsigned int k = 0; k < nv; k++) {
    
    // vertex fit at T=inf
    double sumwz = 0;
    double sumwt = 0;
    double sumw = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double w = tks.pi_[i] * tks.errsum_[i];
      sumwz += w * tks.z_[i];
      sumwt += w * tks.t_[i];
      sumw += w;
    }
    y.z_[k] = sumwz / sumw;
    y.t_[k] = sumwt / sumw;
    
    // estimate Tcrit, eventually do this in the same loop
    double a = 0, b = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double dx = tks.z_[i] - y.z_[k];
      double dt = tks.t_[i] - y.t_[k];
      double w = tks.pi_[i] * tks.errsum_[i];
      a += w * std::pow(dx, 2) * tks.dz2_[i] + std::pow(dt,2) * tks.dt2_[i];
      b += w;
    }
    double Tc = 2. * a / b; // the critical temperature of this vertex
    if (Tc > T0) T0 = Tc;
  }// vertex loop (normally there should be only one vertex at beta=0)
  
#ifdef VI_DEBUG
  if(verbose_){
    std::cout << "DAClustrizerInZT_vect.beta0:   Tc = " << T0 << std::endl;
    int coolingsteps =  1 - int(std::log(T0 * betamax) / std::log(coolingFactor_));
    std::cout << "DAClustrizerInZT_vect.beta0:   nstep = " << coolingsteps << std::endl;
  }
#endif


  if (T0 > 1. / betamax) {
    return betamax / std::pow(coolingFactor_, int(std::log(T0 * betamax) / std::log(coolingFactor_)) - 1);
  } else {
    // ensure at least one annealing step
    return betamax * coolingFactor_;
  }
}


  
bool 
DAClusterizerInZT_vect::split(const double beta,  track_t &tks, vertex_t & y, double threshold ) const{
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split
  
  constexpr double epsilonz=1e-3;      // minimum split size z
  constexpr double epsilont=1e-2;      // minimum split size t
  unsigned int nv = y.getSize();
  
  // avoid left-right biases by splitting highest Tc first
  
  std::vector<std::pair<double, unsigned int> > critical;
  for(unsigned int k=0; k<nv; k++){    
    double Tc= 2*y.swE_[k]/y.sw_[k];
    if (beta*Tc > threshold){
      critical.push_back( make_pair(Tc, k));
    }
  }
  if (critical.empty()) return false;


  std::stable_sort(critical.begin(), critical.end(), std::greater<std::pair<double, unsigned int> >() );
  
  
  bool split=false;
  const unsigned int nt = tks.getSize();

  for(unsigned int ic=0; ic<critical.size(); ic++){
    unsigned int k=critical[ic].second;

    // estimate subcluster positions and weight
    double p1=0, z1=0, t1=0, w1=0;
    double p2=0, z2=0, t2=0, w2=0;
    for(unsigned int i=0; i<nt; ++i){
      if (tks.Z_sum_[i] > 1.e-100) {
	// winner-takes-all, usually overestimates splitting
	double tl = tks.z_[i] < y.z_[k] ? 1.: 0.;
	double tr = 1. - tl;

	// soften it, especially at low T
	/*
	double arg = ( tks.z_[i] - y.z_[k] ) * sqrt(beta * tks.dz2_[i]); // + std::fabs(tks.t_[i] - y.t_[k])
	std::cout << arg << std::endl;
	if(std::fabs(arg) < 20){
	  double t = local_exp(-arg);
	  tl = t/(t+1.);
	  tr = 1/(t+1.);
	}
	*/

	double p = y.pk_[k] * tks.pi_[i] * local_exp(-beta * Eik(tks.z_[i], y.z_[k], tks.dz2_[i], 
                                                                 tks.t_[i], y.t_[k], tks.dt2_[i])) / tks.Z_sum_[i];
	double w = p*tks.errsum_[i];
	p1 += p*tl;  z1 += w*tl*tks.z_[i]; t1 += w*tl*tks.t_[i]; w1 += w*tl;
	p2 += p*tr;  z2 += w*tr*tks.z_[i]; t2 += w*tr*tks.t_[i]; w2 += w*tr;
      }
    }

    if(w1>0){z1 = z1/w1; t1 = t1/w1;} else {z1=y.z_[k]-epsilonz; t1=y.t_[k]-epsilont;}
    if(w2>0){z2 = z2/w2; t2 = t2/w2;} else {z2=y.z_[k]+epsilonz; t2=y.t_[k]+epsilont;}
    
    // reduce split size if there is not enough room
    if( ( k   > 0 ) && ( z1 < (0.6*y.z_[k] + 0.4*y.z_[k-1])) ){ 
      z1 = 0.5*y.z_[k] + 0.5*y.z_[k-1]; 
      t1 = 0.5*y.t_[k] + 0.5*y.t_[k-1]; 
    }
    if( ( k+1 < nv) && ( z2 > (0.6*y.z_[k] + 0.4*y.z_[k+1])) ){ 
      z2 = 0.5*y.z_[k] + 0.5*y.z_[k+1]; 
      t2 = 0.5*y.t_[k] + 0.5*y.t_[k+1]; 
    }
    
#ifdef VI_DEBUG
    if(verbose_){
      if (std::fabs(y.z_[k] - zdumpcenter_) < zdumpwidth_){
	std::cout << " T= " << std::setw(8) << 1./beta 
		  << " Tc= " << critical[ic].first 
		  << "    splitting " << std::fixed << std::setprecision(4) << y.z_[k] 
		  << " --> (" << z1 << ',' << t1<< "),(" << z2 << ',' << t2 
		  << ")     [" << p1 << "," << p2 << "]" ;
	if (std::fabs(z2-z1) > epsilonz || std::fabs(t2-t1) > epsilont){
	  std::cout << std::endl;
	}else{
	  std::cout <<  "  rejected " << std::endl;
	}
      }
    }
#endif

    // split if the new subclusters are significantly separated
    if( std::fabs(z2-z1) > epsilonz || std::fabs(t2-t1) > epsilont){
      split = true;
      double pk1 = p1*y.pk_[k]/(p1+p2);
      double pk2 = p2*y.pk_[k]/(p1+p2);
      y.z_[k]  =  z2;
      y.t_[k]  =  t2;
      y.pk_[k] = pk2;
      y.insertItem(k, z1, t1, pk1);
      nv++;

     // adjust remaining pointers
      for(unsigned int jc=ic; jc < critical.size(); jc++){
        if (critical[jc].second > k) {critical[jc].second++;}
      }
    }
  }
  return split;
}



void DAClusterizerInZT_vect::splitAll( vertex_t & y) const {

  const unsigned int nv = y.getSize();
  
  constexpr double epsilonz = 1e-3; // split all single vertices by 10 um
  constexpr double epsilont = 1e-2; // split all single vertices by 10 ps
  constexpr double zsep = 2 * epsilonz; // split vertices that are isolated by at least zsep (vertices that haven't collapsed)
  constexpr double tsep = 2 * epsilont; 
  vertex_t y1;
  
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout << "Before Split "<< std::endl;
    y.DebugOut();
  }
#endif

  for (unsigned int k = 0; k < nv; k++) {
    if (
	( ( (k == 0)       	|| ( y.z_[k - 1]	< (y.z_[k] - zsep)) ) &&
          ( ((k + 1) == nv)	|| ( y.z_[k + 1] 	> (y.z_[k] + zsep)) )    ) )
      {
	// isolated prototype, split
	double new_z = y.z[k] - epsilonz;
        double new_t = y.t[k] - epsilont;
	y.z[k] = y.z[k] + epsilonz;
        y.t[k] = y.t[k] + epsilont;
	
	double new_pk = 0.5 * y.pk[k];
	y.pk[k] = 0.5 * y.pk[k];
	
	y1.addItem(new_z, new_t, new_pk);
	y1.addItem(y.z_[k], y.t_[k], y.pk_[k]);	
      }
    else if ( (y1.getSize() == 0 ) ||
	      (y1.z_[y1.getSize() - 1] <  (y.z_[k] - zsep)  ) ||
              (y1.t_[y1.getSize() - 1] <  (y.t_[k] - tsep)  ))
      {
	y1.addItem(y.z_[k], y.t_[k], y.pk_[k]);
      }
    else
      {
	y1.z_[y1.getSize() - 1] = y1.z_[y1.getSize() - 1] - epsilonz;
        y1.t_[y1.getSize() - 1] = y1.t_[y1.getSize() - 1] - epsilont;
	y.z_[k] = y.z_[k] + epsilonz;
        y.t_[k] = y.t_[k] + epsilont;
	y1.addItem( y.z_[k], y.t_[k] , y.pk_[k]);
      }
  }// vertex loop

  y = y1;
  y.extractRaw();
  
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout << "After split " << std::endl;
    y.DebugOut();
  }
#endif
}



vector<TransientVertex> 
DAClusterizerInZT_vect::vertices(const vector<reco::TransientTrack> & tracks, const int verbosity) const {
  track_t && tks = fill(tracks);
  tks.extractRaw();
  
  unsigned int nt = tks.getSize();
  double rho0 = 0.0; // start with no outlier rejection
  
  vector<TransientVertex> clusters;
  if (tks.getSize() == 0) return clusters;
  
  vertex_t y; // the vertex prototypes
  
  // initialize:single vertex at infinite temperature
  y.addItem( 0, 0, 1.0);
  
  int niter = 0; // number of iterations
  
  
  // estimate first critical temperature
  double beta = beta0(betamax_, tks, y);
#ifdef VI_DEBUG
  if ( verbose_) std::cout << "Beta0 is " << beta << std::endl;
#endif
  
  niter = 0;
  while ((update(beta, tks, y, false, rho0) > 1.e-6) && 
	 (niter++ < maxIterations_)) {}

  // annealing loop, stop when T<minT  (i.e. beta>1/minT)

  double betafreeze = betamax_ * sqrt(coolingFactor_);

  while (beta < betafreeze) {
    if(useTc_){
      update(beta, tks,y, false, rho0);
      while(merge(y, beta)){update(beta, tks, y, false, rho0);}
      split(beta, tks, y);
      beta=beta/coolingFactor_;
    }else{
      beta=beta/coolingFactor_;
      splitAll(y);
    }
    
    // make sure we are not too far from equilibrium before cooling further
    niter = 0;
    while ((update(beta, tks, y, false, rho0) > 1.e-6) && 
	   (niter++ < maxIterations_)) {}

    if(verbose_){ dump( beta, y, tks, 0); }
  }
  

  if(useTc_){
    //last round of splitting, make sure no critical clusters are left
#ifdef VI_DEBUG
    if(verbose_){ std::cout << "last spliting at " << 1./beta << std::endl; }
#endif
    update(beta, tks,y, false, rho0);// make sure Tc is up-to-date
    while(merge(y,beta)){update(beta, tks,y, false, rho0);}
    unsigned int ntry=0;
    double threshold = 1.0;
    while( split(beta, tks, y, threshold) && (ntry++<10) ){
      niter=0; 
      while((update(beta, tks,y, false, rho0)>1.e-6)  && (niter++ < maxIterations_)){}
      while(merge(y,beta)){update(beta, tks,y, false, rho0);}
#ifdef VI_DEBUG
      if(verbose_){ 
	std::cout << "after final splitting,  try " <<  ntry << std::endl; 
	dump(beta, y, tks, 2); 
      }
#endif
      // relax splitting a bit to reduce multiple split-merge cycles of the same cluster
      threshold *= 1.1; 
    }
  }else{
    // merge collapsed clusters 
    while(merge(y,beta)){update(beta, tks,y, false, rho0);}  
  }
  
#ifdef VI_DEBUG
  if (verbose_) {
    update(beta, tks,y, false, rho0);
    std::cout  << "dump after 1st merging " << endl;
    dump(beta, y, tks, 2);
  }
#endif
  
  
  // switch on outlier rejection at T=minT
  if(dzCutOff_ > 0){
    rho0 = 1./nt;
    for(unsigned int a=0; a<10; a++){ update(beta, tks, y, true, a*rho0/10);} // adiabatic turn-on
  }

  niter=0;
  while ((update(beta, tks, y, true, rho0) > 1.e-8) && (niter++ < maxIterations_)) {};
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout  << "dump after noise-suppression, rho0=" << rho0  << "  niter = " << niter << endl;
    dump(beta, y, tks, 2);
  }
#endif

  // merge again  (some cluster split by outliers collapse here)
  while (merge(y, beta)) {update(beta, tks, y, true, rho0); }
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout  << "dump after merging " << endl;
    dump(beta, y, tks, 2);
  }
#endif

  // go down to the purging temperature (if it is lower than tmin)
  while( beta < betapurge_ ){
    beta = min( beta/coolingFactor_, betapurge_);
    niter = 0;
    while ((update(beta, tks, y, false, rho0) > 1.e-8) && (niter++ < maxIterations_)) {}
  }


  // eliminate insigificant vertices, this is more restrictive at higher T
  while (purge(y, tks, rho0, beta)) {
    niter = 0;
    while (( update(beta, tks, y, true, rho0) >  2.5e-7 * y.getSize() ) && (niter++ < maxIterations_)) {
    }
  }

#ifdef VI_DEBUG
  if (verbose_) {
    update(beta, tks,y, true, rho0);
    std::cout  << " after purging " << std:: endl;
    dump(beta, y, tks, 2);
  }
#endif

  // optionally cool some more without doing anything, to make the assignment harder
  while( beta < betastop_ ){
    beta = min( beta/coolingFactor_, betastop_);
    niter =0;
    while ((update(beta, tks, y, true, rho0) > 1.e-8) && (niter++ < maxIterations_)) {}
  }

#ifdef VI_DEBUG
  if (verbose_) {
    std::cout  << "Final result, rho0=" << std::scientific << rho0 << endl;
    dump(beta, y, tks, 2);
  }
#endif

  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);
  
  // ensure correct normalization of probabilities, should makes double assignment reasonably impossible
  const unsigned int nv = y.getSize();
  for (unsigned int k = 0; k < nv; k++)
     if ( edm::isNotFinite(y.pk_[k]) || edm::isNotFinite(y.z_[k]) ) { y.pk_[k]=0; y.z_[k]=0;}

  for (unsigned int i = 0; i < nt; i++) // initialize
    tks.Z_sum_[i] = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);

  // improve vectorization (does not require reduction ....)
  for (unsigned int k = 0; k < nv; k++) {
     for (unsigned int i = 0; i < nt; i++)  
       tks.Z_sum_[i] += y.pk_[k] * local_exp(-beta * Eik(tks.z_[i], y.z_[k],tks.dz2_[i], tks.t_[i], y.t_[k],tks.dt2_[i]));
  }


  for (unsigned int k = 0; k < nv; k++) {
    GlobalPoint pos(0, 0, y.z_[k]);
    
    vector<reco::TransientTrack> vertexTracks;
    for (unsigned int i = 0; i < nt; i++) {
      if (tks.Z_sum_[i] > 1e-100) {
	
	double p = y.pk_[k] * local_exp(-beta * Eik(tks.z_[i], y.z_[k], tks.dz2_[i],
                                                    tks.t_[i], y.t_[k], tks.dt2_[i] )) / tks.Z_sum_[i];
	if ((tks.pi_[i] > 0) && (p > mintrkweight_)) {
	  vertexTracks.push_back(*(tks.tt[i]));
	  tks.Z_sum_[i] = 0; // setting Z=0 excludes double assignment
	}
      }
    }
    TransientVertex v(pos, dummyError, vertexTracks, 0);
    clusters.push_back(v);
  }

  return clusters;

}

vector<vector<reco::TransientTrack> > DAClusterizerInZT_vect::clusterize(
		const vector<reco::TransientTrack> & tracks) const {
  
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout  << "###################################################" << endl;
    std::cout  << "# vectorized DAClusterizerInZT_vect::clusterize   nt=" << tracks.size() << endl;
    std::cout  << "###################################################" << endl;
  }
#endif
  
  vector<vector<reco::TransientTrack> > clusters;
  vector<TransientVertex> && pv = vertices(tracks);
  
#ifdef VI_DEBUG
  if (verbose_) {
    std::cout  << "# DAClusterizerInZT::clusterize   pv.size=" << pv.size()
					     << endl;
  }
#endif

  if (pv.empty()) {
    return clusters;
  }
  
  // fill into clusters and merge
  vector<reco::TransientTrack> aCluster = pv.begin()->originalTracks();
  
  for (auto k = pv.begin() + 1; k != pv.end(); k++) {
    if ( std::abs(k->position().z() - (k - 1)->position().z()) > (2 * vertexSize_)) {
      // close a cluster
      if (aCluster.size()>1){
	clusters.push_back(aCluster);
      }else{
#ifdef VI_DEBUG
	if(verbose_){
	  std::cout << " one track cluster at " << k->position().z() << "  suppressed" << std::endl;
	}
#endif
      }
      aCluster.clear();
    }
    for (unsigned int i = 0; i < k->originalTracks().size(); i++) {
      aCluster.push_back(k->originalTracks()[i]);
    }
    
  }
  clusters.emplace_back(std::move(aCluster));
  
  return clusters;

}



void DAClusterizerInZT_vect::dump(const double beta, const vertex_t & y,
				  const track_t & tks, int verbosity) const {
#ifdef VI_DEBUG
	const unsigned int nv = y.getSize();
	const unsigned int nt = tks.getSize();
	
	std::vector< unsigned int > iz;
	for(unsigned int j=0; j<nt; j++){ iz.push_back(j); }
	std::sort(iz.begin(), iz.end(), [tks](unsigned int a, unsigned int b){ return tks.z_[a]<tks.z_[b];} ); 
	std::cout  << std::endl;
	std::cout  << "-----DAClusterizerInZT::dump ----" <<  nv << "  clusters " << std::endl;
	std::cout  << "                                                                z= ";
	std::cout  << setprecision(4);
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
	  if (std::fabs(y.z_[ivertex]-zdumpcenter_) < zdumpwidth_){
		std::cout  << setw(8) << fixed << y.z_[ivertex];
	  }
	}
	std::cout  << endl << "T=" << setw(15) << 1. / beta
		   << " Tmin =" << setw(10) << 1./betamax_
			<< "                             Tc= ";
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
	  if (std::fabs(y.z_[ivertex]-zdumpcenter_) < zdumpwidth_){
	    double Tc = 2*y.swE_[ivertex]/y.sw_[ivertex];
	    std::cout  << setw(8) << fixed << setprecision(1) <<  Tc;
	  }
	}
	std::cout  << endl;

	std::cout  <<  "                                                               pk= ";
	double sumpk = 0;
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		sumpk += y.pk_[ivertex];
		if  (std::fabs(y.z_[ivertex] - zdumpcenter_) > zdumpwidth_) continue;
		std::cout  << setw(8) << setprecision(4) << fixed << y.pk_[ivertex];
	}
	std::cout  << endl;

	std::cout << "                                                               nt= ";
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		sumpk += y.pk_[ivertex];
		if  (std::fabs(y.z_[ivertex] - zdumpcenter_) > zdumpwidth_) continue;
		std::cout  << setw(8) << setprecision(1) << fixed << y.pk_[ivertex]*nt;
	}
	std::cout  << endl;

	if (verbosity > 0) {
		double E = 0, F = 0;
		std::cout  << endl;
		std::cout 
		<< "----        z +/- dz                ip +/-dip       pt    phi  eta    weights  ----"
		<< endl;
		std::cout  << setprecision(4);
		for (unsigned int i0 = 0; i0 < nt; i0++) {
		  unsigned int i = iz[i0];
			if (tks.Z_sum_[i] > 0) {
				F -= std::log(tks.Z_sum_[i]) / beta;
			}
			double tz = tks.z_[i];

			if( std::fabs(tz - zdumpcenter_) > zdumpwidth_) continue;
			std::cout  << setw(4) << i << ")" << setw(8) << fixed << setprecision(4)
			 << tz << " +/-" << setw(6) << sqrt(1./tks.dz2_[i]);

			if (tks.tt[i]->track().quality(reco::TrackBase::highPurity)) {
				std::cout  << " *";
			} else {
				std::cout  << "  ";
			}
			if (tks.tt[i]->track().hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)) {
				std::cout  << "+";
			} else {
				std::cout  << "-";
			}
			std::cout  << setw(1)
			 << tks.tt[i]->track().hitPattern().pixelBarrelLayersWithMeasurement(); // see DataFormats/TrackReco/interface/HitPattern.h
			std::cout  << setw(1)
			 << tks.tt[i]->track().hitPattern().pixelEndcapLayersWithMeasurement();
			std::cout  << setw(1) << hex
					<< tks.tt[i]->track().hitPattern().trackerLayersWithMeasurement()
					- tks.tt[i]->track().hitPattern().pixelLayersWithMeasurement()
					<< dec;
			std::cout  << "=" << setw(1) << hex
					<< tks.tt[i]->track().hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS)
					<< dec;

			Measurement1D IP =
					tks.tt[i]->stateAtBeamLine().transverseImpactParameter();
			std::cout  << setw(8) << IP.value() << "+/-" << setw(6) << IP.error();
			std::cout  << " " << setw(6) << setprecision(2)
			 << tks.tt[i]->track().pt() * tks.tt[i]->track().charge();
			std::cout  << " " << setw(5) << setprecision(2)
			 << tks.tt[i]->track().phi() << " " << setw(5)
			 << setprecision(2) << tks.tt[i]->track().eta();

			double sump = 0.;
			for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
			  if  (std::fabs(y.z_[ivertex]-zdumpcenter_) > zdumpwidth_) continue;

				if ((tks.pi_[i] > 0) && (tks.Z_sum_[i] > 0)) {
					//double p=pik(beta,tks[i],*k);
                                  double p = y.pk_[ivertex]  * exp(-beta * Eik(tks.z_[i], y.z_[ivertex], tks.dz2_[i],
                                                                               tks.t_[i], y.t_[ivertex], tks.dt2_[i] )) / tks.Z_sum_[i];
					if (p > 0.0001) {
						std::cout  << setw(8) << setprecision(3) << p;
					} else {
						std::cout  << "    .   ";
					}
					E += p * Eik(tks.z_[i], y.z_[ivertex], tks.dz2_[i],
                                                     tks.t_[i], y.t_[ivertex], tks.dt2_[i] );
					sump += p;
				} else {
					std::cout  << "        ";
				}
			}
			std::cout  << endl;
		}
		std::cout  << endl << "T=" << 1 / beta << " E=" << E << " n=" << y.getSize()
			 << "  F= " << F << endl << "----------" << endl;
	}
#endif
}
