#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ_vect.h"
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

DAClusterizerInZ_vect::DAClusterizerInZ_vect(const edm::ParameterSet& conf) {

  // hardcoded parameters
  maxIterations_ = 100;
  mintrkweight_ = 0.5; // conf.getParameter<double>("mintrkweight");


  // configurable debug outptut debug output
  verbose_ = conf.getUntrackedParameter<bool> ("verbose", false);
  zdumpcenter_ = conf.getUntrackedParameter<double> ("zdumpcenter", 0.);
  zdumpwidth_ = conf.getUntrackedParameter<double> ("zdumpwidth", 20.);
  
  // configurable parameters
  double Tmin = conf.getParameter<double> ("Tmin");
  double Tpurge = conf.getParameter<double> ("Tpurge");
  double Tstop = conf.getParameter<double> ("Tstop");
  vertexSize_ = conf.getParameter<double> ("vertexSize");
  coolingFactor_ = conf.getParameter<double> ("coolingFactor");
  useTc_=true;
  if(coolingFactor_<0){
    coolingFactor_=-coolingFactor_; 
    useTc_=false;
  }
  d0CutOff_ = conf.getParameter<double> ("d0CutOff");
  dzCutOff_ = conf.getParameter<double> ("dzCutOff");
  uniquetrkweight_ = conf.getParameter<double>("uniquetrkweight");
  zmerge_ = conf.getParameter<double>("zmerge");

  if(verbose_){
    std::cout << "DAClusterizerinZ_vect: mintrkweight = " << mintrkweight_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: uniquetrkweight = " << uniquetrkweight_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: zmerge = " << zmerge_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tmin = " << Tmin << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tpurge = " << Tpurge << std::endl;
    std::cout << "DAClusterizerinZ_vect: Tstop = " << Tstop << std::endl;
    std::cout << "DAClusterizerinZ_vect: vertexSize = " << vertexSize_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: coolingFactor = " << coolingFactor_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: d0CutOff = " << d0CutOff_ << std::endl;
    std::cout << "DAClusterizerinZ_vect: dzCutOff = " << dzCutOff_ << std::endl;
  }


  if (Tmin == 0) {
    edm::LogWarning("DAClusterizerinZ_vectorized") << "DAClusterizerInZ: invalid Tmin" << Tmin
						   << "  reset do default " << 1. / betamax_;
  } else {
    betamax_ = 1. / Tmin;
  }


  if ((Tpurge > Tmin) || (Tpurge == 0)) {
    edm::LogWarning("DAClusterizerinZ_vectorized") << "DAClusterizerInZ: invalid Tpurge" << Tpurge
						   << "  set to " << Tmin;
    Tpurge =  Tmin;
  }
  betapurge_ = 1./Tpurge;
  

  if ((Tstop > Tpurge) || (Tstop == 0)) {
    edm::LogWarning("DAClusterizerinZ_vectorized") << "DAClusterizerInZ: invalid Tstop" << Tstop
						   << "  set to  " << max(1., Tpurge);
    Tstop = max(1., Tpurge) ;
  }
  betastop_ = 1./Tstop;
  
}


namespace {
  inline double local_exp( double const& inp) {
    return vdt::fast_exp( inp );
  }
  
  inline void local_exp_list( double const * __restrict__ arg_inp, double * __restrict__ arg_out, const int arg_arr_size) {
    for(int i=0; i!=arg_arr_size; ++i ) arg_out[i]=vdt::fast_exp(arg_inp[i]);
  }

}

//todo: use r-value possibility of c++11 here
DAClusterizerInZ_vect::track_t 
DAClusterizerInZ_vect::fill(const vector<reco::TransientTrack> & tracks) const {

  // prepare track data for clustering
  track_t tks;
  for (auto it = tracks.begin(); it!= tracks.end(); it++){
    if (!(*it).isValid()) continue;
    double t_pi=1.;
    double t_z = ((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    if (std::fabs(t_z) > 1000.) continue;
    auto const & t_mom = (*it).stateAtBeamLine().trackStateAtPCA().momentum();
    //  get the beam-spot
    reco::BeamSpot beamspot = (it->stateAtBeamLine()).beamSpot();
    double t_dz2 = 
      std::pow((*it).track().dzError(), 2) // track errror
      + (std::pow(beamspot.BeamWidthX()*t_mom.x(),2)+std::pow(beamspot.BeamWidthY()*t_mom.y(),2))*std::pow(t_mom.z(),2)/std::pow(t_mom.perp2(),2) // beam spot width
      + std::pow(vertexSize_, 2); // intrinsic vertex size, safer for outliers and short lived decays
    t_dz2 = 1./ t_dz2;
    if (edm::isNotFinite(t_dz2) || t_dz2 < std::numeric_limits<double>::min() ) continue;
    if (d0CutOff_ > 0) {
      Measurement1D atIP =
	(*it).stateAtBeamLine().transverseImpactParameter();// error contains beamspot
      t_pi = 1. / (1. + local_exp(std::pow(atIP.value() / atIP.error(), 2) - std::pow(d0CutOff_, 2))); // reduce weight for high ip tracks
      if (edm::isNotFinite(t_pi) ||  t_pi < std::numeric_limits<double>::epsilon())  continue; // usually is > 0.99
    }
    LogTrace("DAClusterizerinZ_vectorized") << t_z <<' '<< t_dz2 <<' '<< t_pi;
    tks.AddItem(t_z, t_dz2, &(*it), t_pi);
  }
  tks.ExtractRaw();
  
  if (verbose_) {
    std::cout << "Track count " << tks.GetSize() << std::endl;
  }
  
  return tks;
}


namespace {
  inline
  double Eik(double t_z, double k_z, double t_dz2) {
    return std::pow(t_z - k_z, 2) * t_dz2;
  }
}

double DAClusterizerInZ_vect::update(double beta, track_t & gtracks,
				     vertex_t & gvertices, bool useRho0, const double & rho0) const {

  //update weights and vertex positions
  // mass constrained annealing without noise
  // returns the squared sum of changes of vertex positions
  
  const unsigned int nt = gtracks.GetSize();
  const unsigned int nv = gvertices.GetSize();
  
  //initialize sums
  double sumpi = 0;
  
  // to return how much the prototype moved
  double delta = 0;
  

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
    const double track_z = tracks._z[itrack];
    const double botrack_dz2 = -beta*tracks._dz2[itrack];

    // auto-vectorized
    for ( unsigned int ivertex = 0; ivertex < nv; ++ivertex) {
      auto mult_res =  track_z - vertices._z[ivertex];
      vertices._ei_cache[ivertex] = botrack_dz2 * ( mult_res * mult_res );
    }
  };
  
  auto kernel_add_Z = [ nv, Z_init ] (vertex_t const& vertices) -> double
    {
      double ZTemp = Z_init;
      for (unsigned int ivertex = 0; ivertex < nv; ++ivertex) {	
	ZTemp += vertices._pk[ivertex] * vertices._ei[ivertex];
      }
      return ZTemp;
    };

  auto kernel_calc_normalization = [ beta, nv ] (const unsigned int track_num,
						  track_t & tks_vec,
						  vertex_t & y_vec ) {
    auto tmp_trk_pi = tks_vec._pi[track_num];
    auto o_trk_Z_sum = 1./tks_vec._Z_sum[track_num];
    auto o_trk_dz2 = tks_vec._dz2[track_num];
    auto tmp_trk_z = tks_vec._z[track_num];
    auto obeta =  -1./beta;
    
    // auto-vectorized
    for (unsigned int k = 0; k < nv; ++k) {
      y_vec._se[k] +=  y_vec._ei[k] * (tmp_trk_pi* o_trk_Z_sum);
      auto w = y_vec._pk[k] * y_vec._ei[k] * (tmp_trk_pi*o_trk_Z_sum *o_trk_dz2);
      y_vec._sw[k]  += w;
      y_vec._swz[k] += w * tmp_trk_z;
      y_vec._swE[k] += w * y_vec._ei_cache[k]*obeta;
    }
  };
  
  
  for (auto ivertex = 0U; ivertex < nv; ++ivertex) {
    gvertices._se[ivertex] = 0.0;
    gvertices._sw[ivertex] = 0.0;
    gvertices._swz[ivertex] = 0.0;
    gvertices._swE[ivertex] = 0.0;
  }
  
  
  
  // loop over tracks
  for (auto itrack = 0U; itrack < nt; ++itrack) {
    kernel_calc_exp_arg(itrack, gtracks, gvertices);
    local_exp_list(gvertices._ei_cache, gvertices._ei, nv);
    
    gtracks._Z_sum[itrack] = kernel_add_Z(gvertices);
    if (edm::isNotFinite(gtracks._Z_sum[itrack])) gtracks._Z_sum[itrack] = 0.0;
    // used in the next major loop to follow
    sumpi += gtracks._pi[itrack];
    
    if (gtracks._Z_sum[itrack] > 1.e-100){
      kernel_calc_normalization(itrack, gtracks, gvertices);
    }
  }
  
  // now update z and pk
  auto kernel_calc_z = [  sumpi, nv, this, useRho0 ] (vertex_t & vertices ) -> double {
    
    double delta=0;
    // does not vectorizes
    for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex ) {
      if (vertices._sw[ivertex] > 0) {
	auto znew = vertices._swz[ ivertex ] / vertices._sw[ ivertex ];
	// prevents from vectorizing if 
	delta += std::pow( vertices._z[ ivertex ] - znew, 2 );
	vertices._z[ ivertex ] = znew;
      }
#ifdef VI_DEBUG
      else {
	edm::LogInfo("sumw") << "invalid sum of weights in fit: " << vertices._sw[ivertex] << endl;
	if (this->verbose_) {
	  std::cout  << " a cluster melted away ?  pk=" << vertices._pk[ ivertex ] << " sumw="
						   << vertices._sw[ivertex] << endl;
	}
      }
#endif
    }

    auto osumpi = 1./sumpi;
    for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex )
      vertices._pk[ ivertex ] = vertices._pk[ ivertex ] * vertices._se[ ivertex ] * osumpi;

    return delta;
  };
  
  delta += kernel_calc_z(gvertices);
  
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

  // merge the smallest distance clusters first
  std::vector<std::pair<double, unsigned int> > critical;
  for (unsigned int k = 0; (k + 1) < nv; k++) {
    if (std::fabs(y._z[k + 1] - y._z[k]) < zmerge_) {
      critical.push_back( make_pair( std::fabs(y._z[k + 1] - y._z[k]), k) );
    }
  }
  if (critical.empty()) return false;

  std::stable_sort(critical.begin(), critical.end(), std::less<std::pair<double, unsigned int> >() );


  for (unsigned int ik=0; ik < critical.size(); ik++){
    unsigned int k = critical[ik].second;
    double rho = y._pk[k]+y._pk[k+1];
    double swE = y._swE[k]+y._swE[k+1]-y._pk[k]*y._pk[k+1] / rho*std::pow(y._z[k+1]-y._z[k],2);
    double Tc = 2*swE / (y._sw[k]+y._sw[k+1]);

    if(Tc*beta < 1){
      if(verbose_){ std::cout << "merging " << y._z[k + 1] << " and " <<  y._z[k] << "  Tc = " << Tc <<  "  sw = "  << y._sw[k]+y._sw[k+1]  <<std::endl;}
      if(rho > 0){
	y._z[k] = (y._pk[k]*y._z[k] + y._pk[k+1]*y._z[k + 1])/rho;
      }else{
	y._z[k] = 0.5 * (y._z[k] + y._z[k + 1]);
      }
      y._pk[k] = rho;
      y._sw[k] += y._sw[k+1];
      y._swE[k] = swE;
      y.RemoveItem(k+1);
      return true;
    }
  }

  return false;
}




bool 
DAClusterizerInZ_vect::purge(vertex_t & y, track_t & tks, double & rho0, const double beta) const {
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

    double pmax = y._pk[k] / (y._pk[k] + rho0 * local_exp(-beta * dzCutOff_* dzCutOff_));
    for (unsigned int i = 0; i < nt; i++) {
      if (tks._Z_sum[i] > 1.e-100) {
	double p = y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k], tks._dz2[i])) / tks._Z_sum[i];
	sump += p;
	if ((p > uniquetrkweight_ * pmax) && (tks._pi[i] > 0)) {
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
      std::cout  << "eliminating prototype at " << std::setw(10) << std::setprecision(4) << y._z[k0] 
		 << " with sump=" << sumpmin
		 << "  rho*nt =" << y._pk[k0]*nt
		 << endl;
    }
    y.RemoveItem(k0);
    return true;
  } else {
    return false;
  }
}




double 
DAClusterizerInZ_vect::beta0(double betamax, track_t const  & tks, vertex_t const & y) const {
  
  double T0 = 0; // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  const unsigned int nt = tks.GetSize();
  const unsigned int nv = y.GetSize();
  
  for (unsigned int k = 0; k < nv; k++) {
    
    // vertex fit at T=inf
    double sumwz = 0;
    double sumw = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double w = tks._pi[i] * tks._dz2[i];
      sumwz += w * tks._z[i];
      sumw += w;
    }
    y._z[k] = sumwz / sumw;
    
    // estimate Tcrit, eventually do this in the same loop
    double a = 0, b = 0;
    for (unsigned int i = 0; i < nt; i++) {
      double dx = tks._z[i] - y._z[k];
      double w = tks._pi[i] * tks._dz2[i];
      a += w * std::pow(dx, 2) * tks._dz2[i];
      b += w;
    }
    double Tc = 2. * a / b; // the critical temperature of this vertex
    if (Tc > T0) T0 = Tc;
  }// vertex loop (normally there should be only one vertex at beta=0)
  
  if(verbose_){
    std::cout << "DAClustrizerInZ_vect.beta0:   Tc = " << T0 << std::endl;
    int coolingsteps =  1 - int(std::log(T0 * betamax) / std::log(coolingFactor_));
    std::cout << "DAClustrizerInZ_vect.beta0:   nstep = " << coolingsteps << std::endl;
  }


  if (T0 > 1. / betamax) {
    return betamax / std::pow(coolingFactor_, int(std::log(T0 * betamax) / std::log(coolingFactor_)) - 1);
  } else {
    // ensure at least one annealing step
    return betamax * coolingFactor_;
  }
}


  
bool 
DAClusterizerInZ_vect::split(const double beta,  track_t &tks, vertex_t & y, double threshold ) const{
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split
  
  double epsilon=1e-3;      // minimum split size
  unsigned int nv = y.GetSize();
  
  // avoid left-right biases by splitting highest Tc first
  
  std::vector<std::pair<double, unsigned int> > critical;
  for(unsigned int k=0; k<nv; k++){
    double Tc= 2*y._swE[k]/y._sw[k];
    if (beta*Tc > threshold){
      critical.push_back( make_pair(Tc, k));
    }
  }
  if (critical.empty()) return false;


  std::stable_sort(critical.begin(), critical.end(), std::greater<std::pair<double, unsigned int> >() );
  
  
  bool split=false;
  const unsigned int nt = tks.GetSize();

  for(unsigned int ic=0; ic<critical.size(); ic++){
    unsigned int k=critical[ic].second;

    // estimate subcluster positions and weight
    double p1=0, z1=0, w1=0;
    double p2=0, z2=0, w2=0;
    for(unsigned int i=0; i<nt; i++){
      if (tks._Z_sum[i] > 1.e-100) {

	// winner-takes-all, usually overestimates splitting
	double tl = tks._z[i] < y._z[k] ? 1.: 0.;
	double tr = 1. - tl;

	 // soften it, especially at low T
	double arg = (tks._z[i] - y._z[k]) * sqrt(beta * tks._dz2[i]);
	if(std::fabs(arg) < 20){
	  double t = local_exp(-arg);
	  tl = t/(t+1.);
	  tr = 1/(t+1.);
	}

	double p = y._pk[k] * tks._pi[i] * local_exp(-beta * Eik(tks._z[i], y._z[k], tks._dz2[i])) / tks._Z_sum[i];
	double w = p*tks._dz2[i];
	p1 += p*tl ; z1 += w*tl*tks._z[i]; w1 += w*tl;
	p2 += p*tr;  z2 += w*tr*tks._z[i]; w2 += w*tr;
      }
    }

    if(w1>0){z1 = z1/w1;} else {z1=y._z[k]-epsilon;}
    if(w2>0){z2 = z2/w2;} else {z2=y._z[k]+epsilon;}

    // reduce split size if there is not enough room
    if( ( k   > 0 ) && ( z1 < (0.6*y._z[k] + 0.4*y._z[k-1])) ){ z1 = 0.6*y._z[k] + 0.4*y._z[k-1]; }
    if( ( k+1 < nv) && ( z2 > (0.6*y._z[k] + 0.4*y._z[k+1])) ){ z2 = 0.6*y._z[k] + 0.4*y._z[k+1]; }

    if(verbose_){
      if (std::fabs(y._z[k] - zdumpcenter_) < zdumpwidth_){
	std::cout << " T= " << std::setw(8) << 1./beta 
		  << " Tc= " << critical[ic].first 
		  << "    splitting " << std::fixed << std::setprecision(4) << y._z[k] 
		  << " --> " << z1 << "," << z2 
		  << "     [" << p1 << "," << p2 << "]" ;
	if (std::fabs(z2-z1)>epsilon){
	  std::cout << std::endl;
	}else{
	  std::cout <<  "  rejected " << std::endl;
	}
      }
    }

    // split if the new subclusters are significantly separated
    if( (z2-z1) > epsilon){
      split = true;
      double pk1 = p1*y._pk[k]/(p1+p2);
      double pk2 = p2*y._pk[k]/(p1+p2);
      y._z[k]  =  z2;
      y._pk[k] = pk2;
      y.InsertItem(k, z1, pk1);
      nv++;

     // adjust remaining pointers
      for(unsigned int jc=ic; jc < critical.size(); jc++){
        if (critical[jc].second > k) {critical[jc].second++;}
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
  
  if (verbose_) {
    std::cout << "Before Split "<< std::endl;
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
  
  if (verbose_) {
    std::cout << "After split " << std::endl;
    y.DebugOut();
  }
}



vector<TransientVertex> 
DAClusterizerInZ_vect::vertices(const vector<reco::TransientTrack> & tracks, const int verbosity) const {
  track_t && tks = fill(tracks);
  tks.ExtractRaw();
  
  unsigned int nt = tks.GetSize();
  double rho0 = 0.0; // start with no outlier rejection
  
  vector<TransientVertex> clusters;
  if (tks.GetSize() == 0) return clusters;
  
  vertex_t y; // the vertex prototypes
  
  // initialize:single vertex at infinite temperature
  y.AddItem( 0, 1.0);
  
  int niter = 0; // number of iterations
  
  
  // estimate first critical temperature
  double beta = beta0(betamax_, tks, y);
  if ( verbose_) std::cout << "Beta0 is " << beta << std::endl;
  
  niter = 0;
  while ((update(beta, tks, y, false, rho0) > 1.e-6) && 
	 (niter++ < maxIterations_)) {}

  // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)

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
    if(verbose_){ std::cout << "last spliting at " << 1./beta << std::endl; }
    update(beta, tks,y, false, rho0);// make sure Tc is up-to-date
    while(merge(y,beta)){update(beta, tks,y, false, rho0);}
    unsigned int ntry=0;
    double threshold = 1.0;
    while( split(beta, tks, y, threshold) && (ntry++<10) ){
      niter=0; 
      while((update(beta, tks,y, false, rho0)>1.e-6)  && (niter++ < maxIterations_)){}
      while(merge(y,beta)){update(beta, tks,y, false, rho0);}
      if(verbose_){ 
	std::cout << "after final splitting,  try " <<  ntry << std::endl; 
	dump(beta, y, tks, 2); 
      }
      // relax splitting a bit to reduce multiple split-merge cycles of the same cluster
      threshold *= 1.1; 
    }
  }else{
    // merge collapsed clusters 
    while(merge(y,beta)){update(beta, tks,y, false, rho0);}  
  }
  
  if (verbose_) {
    update(beta, tks,y, false, rho0);
    std::cout  << "dump after 1st merging " << endl;
    dump(beta, y, tks, 2);
  }
  
  
  // switch on outlier rejection at T=Tmin
  if(dzCutOff_ > 0){
    rho0 = 1./nt;
    for(unsigned int a=0; a<10; a++){ update(beta, tks, y, true, a*rho0/10);} // adiabatic turn-on
  }

  niter=0;
  while ((update(beta, tks, y, true, rho0) > 1.e-8) && (niter++ < maxIterations_)) {};
  if (verbose_) {
    std::cout  << "dump after noise-suppression, rho0=" << rho0  << "  niter = " << niter << endl;
    dump(beta, y, tks, 2);
  }

  // merge again  (some cluster split by outliers collapse here)
  while (merge(y, beta)) {update(beta, tks, y, true, rho0); }
  if (verbose_) {
    std::cout  << "dump after merging " << endl;
    dump(beta, y, tks, 2);
  }

  // go down to the purging temperature (if it is lower than tmin)
  while( beta < betapurge_ ){
    beta = min( beta/coolingFactor_, betapurge_);
    niter = 0;
    while ((update(beta, tks, y, false, rho0) > 1.e-8) && (niter++ < maxIterations_)) {}
  }


  // eliminate insigificant vertices, this is more restrictive at higher T
  while (purge(y, tks, rho0, beta)) {
    niter = 0;
    while ((update(beta, tks, y, true, rho0) > 1.e-6) && (niter++ < maxIterations_)) {}
  }

  if (verbose_) {
    update(beta, tks,y, true, rho0);
    std::cout  << " after purging " << std:: endl;
    dump(beta, y, tks, 2);
  }

  // optionally cool some more without doing anything, to make the assignment harder
  while( beta < betastop_ ){
    beta = min( beta/coolingFactor_, betastop_);
    niter =0;
    while ((update(beta, tks, y, true, rho0) > 1.e-8) && (niter++ < maxIterations_)) {}
  }


  if (verbose_) {
    std::cout  << "Final result, rho0=" << std::scientific << rho0 << endl;
    dump(beta, y, tks, 2);
  }

  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);
  
  // ensure correct normalization of probabilities, should makes double assignment reasonably impossible
  const unsigned int nv = y.GetSize();
  for (unsigned int k = 0; k < nv; k++)
     if ( edm::isNotFinite(y._pk[k]) || edm::isNotFinite(y._z[k]) ) { y._pk[k]=0; y._z[k]=0;}

  for (unsigned int i = 0; i < nt; i++) // initialize
    tks._Z_sum[i] = rho0 * local_exp(-beta * dzCutOff_ * dzCutOff_);

  // improve vectorization (does not require reduction ....)
  for (unsigned int k = 0; k < nv; k++) {
     for (unsigned int i = 0; i < nt; i++)  
      tks._Z_sum[i] += y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],tks._dz2[i]));
  }


  for (unsigned int k = 0; k < nv; k++) {
    GlobalPoint pos(0, 0, y._z[k]);
    
    vector<reco::TransientTrack> vertexTracks;
    for (unsigned int i = 0; i < nt; i++) {
      if (tks._Z_sum[i] > 1e-100) {
	
	double p = y._pk[k] * local_exp(-beta * Eik(tks._z[i], y._z[k],
						    tks._dz2[i])) / tks._Z_sum[i];
	if ((tks._pi[i] > 0) && (p > mintrkweight_)) {
	  vertexTracks.push_back(*(tks.tt[i]));
	  tks._Z_sum[i] = 0; // setting Z=0 excludes double assignment
	}
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
  vector<TransientVertex> && pv = vertices(tracks);
  
  if (verbose_) {
    std::cout  << "# DAClusterizerInZ::clusterize   pv.size=" << pv.size()
					     << endl;
  }
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
	if(verbose_){
	  std::cout << " one track cluster at " << k->position().z() << "  suppressed" << std::endl;
	}
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



void DAClusterizerInZ_vect::dump(const double beta, const vertex_t & y,
		const track_t & tks, int verbosity) const {

	const unsigned int nv = y.GetSize();
	const unsigned int nt = tks.GetSize();
	
	std::vector< unsigned int > iz;
	for(unsigned int j=0; j<nt; j++){ iz.push_back(j); }
	std::sort(iz.begin(), iz.end(), [tks](unsigned int a, unsigned int b){ return tks._z[a]<tks._z[b];} ); 
	std::cout  << std::endl;
	std::cout  << "-----DAClusterizerInZ::dump ----" <<  nv << "  clusters " << std::endl;
	std::cout  << "                                                                z= ";
	std::cout  << setprecision(4);
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
	  if (std::fabs(y._z[ivertex]-zdumpcenter_) < zdumpwidth_){
		std::cout  << setw(8) << fixed << y._z[ivertex];
	  }
	}
	std::cout  << endl << "T=" << setw(15) << 1. / beta
		   << " Tmin =" << setw(10) << 1./betamax_
			<< "                             Tc= ";
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
	  if (std::fabs(y._z[ivertex]-zdumpcenter_) < zdumpwidth_){
	    double Tc = 2*y._swE[ivertex]/y._sw[ivertex];
	    std::cout  << setw(8) << fixed << setprecision(1) <<  Tc;
	  }
	}
	std::cout  << endl;

	std::cout  <<  "                                                               pk= ";
	double sumpk = 0;
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		sumpk += y._pk[ivertex];
		if  (std::fabs(y._z[ivertex] - zdumpcenter_) > zdumpwidth_) continue;
		std::cout  << setw(8) << setprecision(4) << fixed << y._pk[ivertex];
	}
	std::cout  << endl;

	std::cout << "                                                               nt= ";
	for (unsigned int ivertex = 0; ivertex < nv; ++ ivertex) {
		sumpk += y._pk[ivertex];
		if  (std::fabs(y._z[ivertex] - zdumpcenter_) > zdumpwidth_) continue;
		std::cout  << setw(8) << setprecision(1) << fixed << y._pk[ivertex]*nt;
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
			if (tks._Z_sum[i] > 0) {
				F -= std::log(tks._Z_sum[i]) / beta;
			}
			double tz = tks._z[i];

			if( std::fabs(tz - zdumpcenter_) > zdumpwidth_) continue;
			std::cout  << setw(4) << i << ")" << setw(8) << fixed << setprecision(4)
			 << tz << " +/-" << setw(6) << sqrt(1./tks._dz2[i]);

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
					<< tks.tt[i]->track().hitPattern().numberOfHits(reco::HitPattern::MISSING_OUTER_HITS)
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
			  if  (std::fabs(y._z[ivertex]-zdumpcenter_) > zdumpwidth_) continue;

				if ((tks._pi[i] > 0) && (tks._Z_sum[i] > 0)) {
					//double p=pik(beta,tks[i],*k);
					double p = y._pk[ivertex]  * exp(-beta * Eik(tks._z[i], y._z[ivertex], tks._dz2[i])) / tks._Z_sum[i];
					if (p > 0.0001) {
						std::cout  << setw(8) << setprecision(3) << p;
					} else {
						std::cout  << "    .   ";
					}
					E += p * Eik(tks._z[i], y._z[ivertex], tks._dz2[i]);
					sump += p;
				} else {
					std::cout  << "        ";
				}
			}
			std::cout  << endl;
		}
		std::cout  << endl << "T=" << 1 / beta << " E=" << E << " n=" << y.GetSize()
			 << "  F= " << F << endl << "----------" << endl;
	}
}
