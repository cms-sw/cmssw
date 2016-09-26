#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "FWCore/Utilities/interface/isFinite.h"

using namespace std;

namespace {
  constexpr double epsilon = 1.0e-3;
  constexpr double vertexSizeTime = 0.008;
  constexpr double dtCutOff = 4.0;

  bool recTrackLessZ1(const DAClusterizerInZT::track_t & tk1,
                      const DAClusterizerInZT::track_t & tk2)
  {
    return tk1.z < tk2.z;
  }

  double sqr(double f) { return f*f; }
}


vector<DAClusterizerInZT::track_t> 
DAClusterizerInZT::fill( const vector<reco::TransientTrack> & tracks )const{
  // prepare track data for clustering
  vector<track_t> tks;
  tks.reserve(tracks.size());
  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
    track_t t;
    t.pi = 1.;
    auto tsPCA = (*it).stateAtBeamLine().trackStateAtPCA();
    t.z = tsPCA.position().z();
    t.t = it->timeExt(); // the time
    
    if (std::abs(t.z) > 1000.) continue;
    auto const & t_mom = tsPCA.momentum();
    //  get the beam-spot
    reco::BeamSpot beamspot = (*it).stateAtBeamLine().beamSpot();
    t.dz2 = 
      sqr((*it).track().dzError()) // track errror
      + (sqr(beamspot.BeamWidthX()*t_mom.x())+sqr(beamspot.BeamWidthY()*t_mom.y()))*sqr(t_mom.z())/sqr(t_mom.perp2()) // beam spot width
      + sqr(vertexSize_); // intrinsic vertex size, safer for outliers and short lived decays
    //t.dz2 = 1./ t_dz2;
    
    t.dtz = 0.;
    t.dt2 =sqr((*it).dtErrorExt()) + sqr(vertexSizeTime); // the ~injected~ timing error, need to add a small minimum vertex size in time
    
    if (d0CutOff_>0){
      Measurement1D IP = (*it).stateAtBeamLine().transverseImpactParameter();// error constains beamspot
      t.pi=1./(1.+std::exp(sqr(IP.value()/IP.error()) - sqr(d0CutOff_)));  // reduce weight for high ip tracks  
    }else{
      t.pi=1.;
    }    
    t.tt=&(*it);
    t.zi=1.;
    if( edm::isFinite(t.pi) &&  t.pi >= std::numeric_limits<double>::epsilon() ) {
      tks.push_back(t);
    }
  }

  tks.shrink_to_fit();
  return tks;
}





double DAClusterizerInZT::e_ik(const track_t & t, const vertex_t &k ) const {
  return sqr(t.z-k.z)/t.dz2 + sqr(t.t - k.t)/t.dt2;
}
  





double DAClusterizerInZT::update( double beta,
                                  vector<track_t> & tks,
                                  vector<vertex_t> & y,
                                  const double rho0 ) const {
  // MVF style, no more vertex weights, update tracks weights and vertex positions, with noise 
  // returns the squared sum of changes of vertex positions

  unsigned int nt=tks.size();

  //initialize sums
  double sumpi = 0.;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    k->sw = 0.;   k->swz = 0.; k->swt = 0.; k->se = 0.;    
    k->swE = 0.;  k->tC=0.;
  }


  // loop over tracks
  for(unsigned int i=0; i<nt; i++){

    // update pik and Zi and Ti
    double Zi = rho0*std::exp(-beta*(dzCutOff_*dzCutOff_));// cut-off (eventually add finite size in time)
    //double Ti = 0.; // dt0*std::exp(-beta*dtCutOff_);
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      k->ei = std::exp(-beta*e_ik(tks[i],*k));// cache exponential for one track at a time
      Zi   += k->pk * k->ei;
    }
    tks[i].zi=Zi;
    sumpi += tks[i].pi;
    
    // normalization
    if (tks[i].zi>0){
      // accumulate weighted z and weights for vertex update
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	k->se += tks[i].pi* k->ei / Zi;
	double w = k->pk * tks[i].pi * k->ei /( Zi * ( tks[i].dz2 * tks[i].dt2 ) );
	k->sw  += w;
	k->swz += w * tks[i].z;
        k->swt += w * tks[i].t;
	k->swE += w * e_ik(tks[i],*k);
      }
    }
  } // end of track loop


  // now update z
  double delta=0;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( k->sw > 0){
      const double znew=k->swz/k->sw; 
      const double tnew=k->swt/k->sw;
      delta += sqr(k->z-znew) + sqr(k->t-tnew);
      k->z   = znew;
      k->t   = tnew;
      k->tC  = 2*k->swE/k->sw;
    }else{
      edm::LogInfo("sumw") <<  "invalid sum of weights in fit: " << k->sw << endl;
      if(verbose_){cout << " a cluster melted away ?  pk=" << k->pk <<  " sumw=" << k->sw <<  endl;}
      k->tC = (rho0 == 0. ? -1 : 0);
    }
    if(rho0 == 0.) k->pk = k->pk * k->se / sumpi;
  }

  // return how much the prototypes moved
  return delta;
}

bool DAClusterizerInZT::merge(vector<vertex_t> & y, int nt)const{
  // merge clusters that collapsed or never separated, return true if vertices were merged, false otherwise
  
  if(y.size()<2)  return false;
  
  for(vector<vertex_t>::iterator k=y.begin(); (k+1)!=y.end(); k++){
    if( std::abs( (k+1)->z - k->z ) < epsilon &&
        std::abs( (k+1)->t - k->t ) < epsilon    ){  // with fabs if only called after freeze-out (splitAll() at highter T)
      double rho = k->pk + (k+1)->pk;
      if(rho>0){
        k->z = ( k->pk * k->z + (k+1)->z * (k+1)->pk)/rho;
        k->t = ( k->pk * k->t + (k+1)->t * (k+1)->pk)/rho;
      }else{
        k->z = 0.5*(k->z + (k+1)->z);
        k->t = 0.5*(k->t + (k+1)->t);
      }
      k->pk = rho;
      
      y.erase(k+1);
      return true;  
    }
  }
  
  return false;
}

bool DAClusterizerInZT::merge(vector<vertex_t> & y, double & beta)const{
  // merge clusters that collapsed or never separated, 
  // only merge if the estimated critical temperature of the merged vertex is below the current temperature
  // return true if vertices were merged, false otherwise
  if(y.size()<2)  return false;

  for(vector<vertex_t>::iterator k=y.begin(); (k+1)!=y.end(); k++){
    if ( std::abs((k+1)->z - k->z) < 2*epsilon && 
         std::abs((k+1)->t - k->t) < 2*epsilon    ) { 
      double rho=k->pk + (k+1)->pk;
      double swE=k->swE+(k+1)->swE - k->pk * (k+1)->pk / rho * ( sqr((k+1)->z - k->z) + 
                                                                 sqr((k+1)->t - k->t)   );
      double Tc=2*swE/(k->sw+(k+1)->sw);
      
      if(Tc*beta<1){
	if(rho>0){
	  k->z = ( k->pk * k->z + (k+1)->z * (k+1)->pk)/rho;
          k->t = ( k->pk * k->t + (k+1)->t * (k+1)->pk)/rho;
	}else{
	  k->z = 0.5*(k->z + (k+1)->z);
          k->t = 0.5*(k->t + (k+1)->t);
	}
	k->pk  = rho;
	k->sw += (k+1)->sw;
	k->swE = swE;
	k->tC  = Tc;
	y.erase(k+1);
	return true; 
      }
    }
  }

  return false;
}

bool DAClusterizerInZT::purge(vector<vertex_t> & y, vector<track_t> & tks, double & rho0, const double beta)const{
  // eliminate clusters with only one significant/unique track
  if(y.size()<2)  return false;
  
  unsigned int nt=tks.size();
  double sumpmin=nt;
  vector<vertex_t>::iterator k0=y.end();
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
    int nUnique=0;
    double sump=0;
    double pmax=k->pk/(k->pk+rho0*exp(-beta*dzCutOff_*dzCutOff_));
    for(unsigned int i=0; i<nt; i++){
      if(tks[i].zi > 0){
	double p = k->pk * std::exp(-beta*e_ik(tks[i],*k)) / tks[i].zi ;
	sump+=p;
	if( (p > 0.9*pmax) && (tks[i].pi>0) ){ nUnique++; }
      }
    }

    if((nUnique<2)&&(sump<sumpmin)){
      sumpmin=sump;
      k0=k;
    }
  }
 
  if(k0!=y.end()){
    if(verbose_){cout << "eliminating prototype at " << k0->z << "," << k0->t 
                      << " with sump=" << sumpmin << endl;}
    y.erase(k0);
    return true;
  }else{
    return false;
  }
}

double DAClusterizerInZT::beta0( double betamax,
                                 vector<track_t> & tks,
                                 vector<vertex_t> & y ) const {
  
  double T0=0;  // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  unsigned int nt=tks.size();

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){

    // vertex fit at T=inf 
    double sumwz=0.;
    double sumwt=0.;
    double sumw=0.;
    for(unsigned int i=0; i<nt; i++){
      double w = tks[i].pi/(tks[i].dz2 * tks[i].dt2);
      sumwz += w*tks[i].z;
      sumwt += w*tks[i].t;
      sumw  += w;
    }
    k->z = sumwz/sumw;
    k->t = sumwt/sumw;

    // estimate Tcrit, eventually do this in the same loop
    double a=0, b=0;
    for(unsigned int i=0; i<nt; i++){
      double dx = tks[i].z-(k->z);
      double dt = tks[i].t-(k->t);
      double w  = tks[i].pi/(tks[i].dz2 * tks[i].dt2);
      a += w*(sqr(dx)/tks[i].dz2 + sqr(dt)/tks[i].dt2);
      b += w;
    }
    double Tc= 2.*a/b;  // the critical temperature of this vertex
    if(Tc>T0) T0=Tc;
  }// vertex loop (normally there should be only one vertex at beta=0)
  
  if (T0>1./betamax){
    return betamax/pow(coolingFactor_, int(std::log(T0*betamax)*logCoolingFactor_)-1 );
  }else{
    // ensure at least one annealing step
    return betamax/coolingFactor_;
  }
}

bool DAClusterizerInZT::split( double beta,
                               vector<track_t> & tks,
                               vector<vertex_t> & y,
                               double threshold ) const {
  // split only critical vertices (Tc >~ T=1/beta   <==>   beta*Tc>~1)
  // an update must have been made just before doing this (same beta, no merging)
  // returns true if at least one cluster was split
  bool split=false;
  
  // avoid left-right biases by splitting highest Tc first
  
  std::vector<std::pair<double, unsigned int> > critical;
  for(unsigned int ik=0; ik<y.size(); ik++){
    if (beta*y[ik].tC > 1.){
      critical.push_back( make_pair(y[ik].tC, ik));
    }
  }
  std::stable_sort(critical.begin(), critical.end(), std::greater<std::pair<double, unsigned int> >() );

  for(unsigned int ic=0; ic<critical.size(); ic++){
    unsigned int ik=critical[ic].second;
    // estimate subcluster positions and weight
    double p1=0, z1=0, t1=0, w1=0;
    double p2=0, z2=0, t2=0, w2=0;
    //double sumpi=0;
    for(unsigned int i=0; i<tks.size(); i++){
      if(tks[i].zi>0){
	//sumpi+=tks[i].pi;
	double p=y[ik].pk * exp(-beta*e_ik(tks[i],y[ik])) / tks[i].zi*tks[i].pi;
	double w=p/(tks[i].dz2 * tks[i].dt2);
	if(tks[i].z < y[ik].z){
	  p1+=p; z1+=w*tks[i].z; t1+=w*tks[i].t; w1+=w;
	}else{
	  p2+=p; z2+=w*tks[i].z; t2+=w*tks[i].t; w2+=w;
	}
      }
    }
    if(w1>0){  z1=z1/w1; t1=t1/w1;} else{ z1=y[ik].z-epsilon; t1=y[ik].t-epsilon; }
    if(w2>0){  z2=z2/w2; t2=t2/w2;} else{ z2=y[ik].z+epsilon; t2=y[ik].t+epsilon;}

    // reduce split size if there is not enough room
    if( ( ik   > 0       ) && ( y[ik-1].z>=z1 ) ){ z1=0.5*(y[ik].z+y[ik-1].z); t1=0.5*(y[ik].t+y[ik-1].t); }
    if( ( ik+1 < y.size()) && ( y[ik+1].z<=z2 ) ){ z2=0.5*(y[ik].z+y[ik+1].z); t2=0.5*(y[ik].t+y[ik+1].t); }

    // split if the new subclusters are significantly separated
    if( std::abs(z2-z1) > epsilon || std::abs(t2-t1) > epsilon ){
      split=true;
      vertex_t vnew;
      vnew.pk = p1*y[ik].pk/(p1+p2);
      y[ik].pk= p2*y[ik].pk/(p1+p2);
      vnew.z  = z1;
      vnew.t  = t1;
      y[ik].z = z2;
      y[ik].t = t2;
      y.insert(y.begin()+ik, vnew);

     // adjust remaining pointers
      for(unsigned int jc=ic; jc<critical.size(); jc++){
	if (critical[jc].second>ik) {critical[jc].second++;}
      }
    }
  }

  //  stable_sort(y.begin(), y.end(), clusterLessZ);
  return split;
}





void DAClusterizerInZT::splitAll( vector<vertex_t> & y ) const {

  constexpr double zsep=2*epsilon;    // split vertices that are isolated by at least zsep (vertices that haven't collapsed)
  constexpr double tsep=2*epsilon;    // check t as well
  
  vector<vertex_t> y1;

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( ( (k==y.begin())|| (k-1)->z < k->z - zsep) && (((k+1)==y.end()  )|| (k+1)->z > k->z + zsep)) { 
      // isolated prototype, split
      vertex_t vnew;
      vnew.z  = k->z - epsilon;
      vnew.t  = k->t - epsilon;
      (*k).z  = k->z + epsilon;
      (*k).t  = k->t + epsilon;
      vnew.pk= 0.5* (*k).pk;
      (*k).pk= 0.5* (*k).pk;
      y1.push_back(vnew);
      y1.push_back(*k);

    }else if( y1.empty() || (y1.back().z < k->z -zsep) || (y1.back().t < k->t - tsep) ){
      y1.push_back(*k);
    }else{
      y1.back().z -= epsilon;
      y1.back().t -= epsilon;
      k->z += epsilon;
      k->t += epsilon;
      y1.push_back(*k);
    }
  }// vertex loop
  
  y=y1;
}
 

DAClusterizerInZT::DAClusterizerInZT(const edm::ParameterSet& conf) :
  verbose_(conf.getUntrackedParameter<bool>("verbose", false)),
  useTc_(true),
  vertexSize_(conf.getParameter<double>("vertexSize")),
  maxIterations_(100),
  coolingFactor_(std::sqrt(conf.getParameter<double>("coolingFactor"))),  
  betamax_(0.1),
  betastop_(1.0),
  dzCutOff_(conf.getParameter<double>("dzCutOff")),
  d0CutOff_(conf.getParameter<double>("d0CutOff")),
  dtCutOff_(dtCutOff)
{

  double Tmin = conf.getParameter<double>("Tmin")*std::sqrt(2.0);// scale up by sqrt(D=2)
  if (Tmin==0){
    edm::LogWarning("DAClusterizerInZT") << "DAClusterizerInZT: invalid Tmin" << Tmin << "  reset do default " << 1./betamax_ << endl;
  }else{
    betamax_ = 1./Tmin;
  }

  // for testing, negative cooling factor: revert to old splitting scheme
  if(coolingFactor_<0){
    coolingFactor_=-coolingFactor_; useTc_=false;
  }
  
  logCoolingFactor_ = 1.0/std::log(coolingFactor_);
}


void DAClusterizerInZT::dump(const double beta, const vector<vertex_t> & y, const vector<track_t> & tks0, int verbosity)const{

  // copy and sort for nicer printout
  vector<track_t> tks; 
  for(vector<track_t>::const_iterator t=tks0.begin(); t!=tks0.end(); t++){tks.push_back(*t); }
  std::stable_sort(tks.begin(), tks.end(), recTrackLessZ1);

  cout << "-----DAClusterizerInZT::dump ----" << endl;
  cout << " beta=" << beta << "   betamax= " << betamax_ << endl;
  cout << "                                                               z= ";
  cout.precision(4);
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(8) << fixed << k->z;
  }
  cout << endl << "                                                               t= ";
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(8) << fixed << k->t;
  }
  cout << endl << "T=" << setw(15) << 1./beta <<"                                             Tc= ";
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(8) << fixed << k->tC ;
  }
 
  cout << endl << "                                                              pk=";
  double sumpk=0;
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout <<  setw(8) <<  setprecision(3) <<  fixed << k->pk;
    sumpk+=k->pk;
  }
  cout  << endl;

  if(verbosity>0){
    double E=0, F=0;
    cout << endl;
    cout << "----       z +/- dz        t +/- dt        ip +/-dip       pt    phi  eta    weights  ----" << endl;
    cout.precision(4);
    for(unsigned int i=0; i<tks.size(); i++){
      if (tks[i].zi>0){	F-=log(tks[i].zi)/beta;}
      double tz= tks[i].z;
      double tt= tks[i].t;
      cout <<  setw (3)<< i << ")" <<  setw (8) << fixed << setprecision(4)<<  tz << " +/-" <<  setw (6)<< sqrt(tks[i].dz2) 
           << setw(8) << fixed << setprecision(4) << tt << " +/-" << setw(6) << std::sqrt(tks[i].dt2)  ;

      if(tks[i].tt->track().quality(reco::TrackBase::highPurity)){ cout << " *";}else{cout <<"  ";}
      if(tks[i].tt->track().hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)){cout <<"+";}else{cout << "-";}
      
      cout << setw(1) << tks[i].tt->track().hitPattern().pixelBarrelLayersWithMeasurement(); // see DataFormats/TrackReco/interface/HitPattern.h
      cout << setw(1) << tks[i].tt->track().hitPattern().pixelEndcapLayersWithMeasurement(); 
      cout << setw(1) << hex << tks[i].tt->track().hitPattern().trackerLayersWithMeasurement()-tks[i].tt->track().hitPattern().pixelLayersWithMeasurement() <<dec; 
      cout << "=" << setw(1)<<hex <<tks[i].tt->track().hitPattern().numberOfHits(reco::HitPattern::MISSING_OUTER_HITS) << dec;

      Measurement1D IP=tks[i].tt->stateAtBeamLine().transverseImpactParameter();
      cout << setw (8) << IP.value() << "+/-" << setw (6) << IP.error();
      cout << " " << setw(6) << setprecision(2)  << tks[i].tt->track().pt()*tks[i].tt->track().charge();
      cout << " " << setw(5) << setprecision(2) << tks[i].tt->track().phi() 
	   << " "  << setw(5)  << setprecision(2)   << tks[i].tt->track().eta() ;

      double sump=0.;
      for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
	if((tks[i].pi>0)&&(tks[i].zi>0)){
	  //double p=pik(beta,tks[i],*k);
	  double p=k->pk * std::exp(-beta*e_ik(tks[i],*k)) / tks[i].zi; 
	  if( p > 0.0001){
	    cout <<  setw (8) <<  setprecision(3) << p;
	  }else{
	    cout << "    .   ";
	  }
	  E+=p*e_ik(tks[i],*k);
	  sump+=p;
	}else{
	    cout << "        ";
	}
      }
      cout << endl;
    }
    cout << endl << "T=" << 1/beta  << " E=" << E << " n="<< y.size() << "  F= " << F <<  endl <<  "----------" << endl;
  }
}





vector< TransientVertex >
DAClusterizerInZT::vertices(const vector<reco::TransientTrack> & tracks, const int verbosity) 
const
{
 
  vector<track_t> tks=fill(tracks);
  unsigned int nt=tks.size();
  double rho0=0.0;  // start with no outlier rejection

  vector< TransientVertex > clusters;
  if (tks.empty()) return clusters;

  vector<vertex_t> y; // the vertex prototypes

  // initialize:single vertex at infinite temperature
  vertex_t vstart;
  vstart.z=0.;
  vstart.t=0.;
  vstart.pk=1.;
  y.push_back(vstart);
  int niter=0;      // number of iterations
  

  // estimate first critical temperature
  double beta=beta0(betamax_, tks, y);
  niter=0; 
  while((update(beta, tks,y)>1.e-6)  && (niter++ < maxIterations_)){ }

  // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)
  while(beta<betamax_){ 

    if(useTc_){
      update(beta, tks,y);
      while(merge(y,beta)){update(beta, tks,y);}
      split(beta, tks,y, 1.);
      beta=beta/coolingFactor_;
    }else{
      beta=beta/coolingFactor_;
      splitAll(y);
    }


   // make sure we are not too far from equilibrium before cooling further
   niter=0; 
   while((update(beta, tks,y)>1.e-6)  && (niter++ < maxIterations_)){ }

  }

  if(useTc_){
    // last round of splitting, make sure no critical clusters are left
    update(beta, tks,y);
    while(merge(y,beta)){update(beta, tks,y);}
    unsigned int ntry=0;
    while( split(beta, tks,y,1.) && (ntry++<10) ){
      niter=0; 
      while((update(beta, tks,y)>1.e-6)  && (niter++ < maxIterations_)){}
      merge(y,beta);
      update(beta, tks,y);
    }
  }else{
    // merge collapsed clusters 
    while(merge(y,beta)){update(beta, tks,y);}  
    if(verbose_ ){ cout << "dump after 1st merging " << endl;  dump(beta,y,tks,2);}
  }
  



  // switch on outlier rejection
  rho0=1./nt; 
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ k->pk =1.; }  // democratic
  niter=0; 
  while( (update(beta, tks,y,rho0) > 1.e-8)  && (niter++ < maxIterations_) ){  }
  if(verbose_  ){ cout << "rho0=" << rho0 <<   " niter=" << niter <<  endl; dump(beta,y,tks,2);}

  
  // merge again  (some cluster split by outliers collapse here)
  while(merge(y,tks.size())){}  
  if(verbose_  ){ cout << "dump after 2nd merging " << endl;  dump(beta,y,tks,2);}


  // continue from freeze-out to Tstop (=1) without splitting, eliminate insignificant vertices
  while(beta<=betastop_){
    while(purge(y,tks,rho0, beta)){
      niter=0; 
      while((update(beta, tks, y, rho0) > 1.e-6)  && (niter++ < maxIterations_)){  }
    } 
    beta/=coolingFactor_;
    niter=0; 
    while((update(beta, tks, y, rho0) > 1.e-6)  && (niter++ < maxIterations_)){  }
  }
  
  if(verbose_){
   cout << "Final result, rho0=" << rho0 << endl;
   dump(beta,y,tks,2);
  }
  
  // ensure correct normalization of probabilities, should make double assginment reasonably impossible
  for(unsigned int i=0; i<nt; i++){  
    tks[i].zi=rho0*exp(-beta*( dzCutOff_*dzCutOff_));
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
      tks[i].zi += k->pk * exp(-beta*e_ik(tks[i],*k));
    }
  }


  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
    GlobalPoint pos(0, 0, k->z);
    double time = k->t;
    vector< reco::TransientTrack > vertexTracks;
    //double max_track_time_err2 = 0;
    double mean = 0.;
    double expv_x2 = 0.;
    double normw = 0.;
    for(unsigned int i=0; i<nt; i++){
      const double invdt = 1.0/std::sqrt(tks[i].dt2);
      if(tks[i].zi>0){
	double p = k->pk * exp(-beta*e_ik(tks[i],*k)) / tks[i].zi;
	if( (tks[i].pi>0) && ( p > 0.5 ) ){ 
          //std::cout << "pushing back " << i << ' ' << tks[i].tt << std::endl;
          vertexTracks.push_back(*(tks[i].tt)); tks[i].zi=0; 
          mean     += tks[i].t*invdt*p;
          expv_x2  += tks[i].t*tks[i].t*invdt*p;
          normw    += invdt*p;
        } // setting Z=0 excludes double assignment        
      }
    }
    mean = mean/normw;
    expv_x2 = expv_x2/normw;
    const double time_var = expv_x2 - mean*mean;    
    const double crappy_error_guess = std::sqrt(time_var);
    GlobalError dummyErrorWithTime(0,
                                   0,0,
                                   0,0,0,
                                   0,0,0,crappy_error_guess);
    TransientVertex v(pos, time, dummyErrorWithTime, vertexTracks, 5);
    clusters.push_back(v);
  }


  return clusters;

}





vector< vector<reco::TransientTrack> >
DAClusterizerInZT::clusterize(const vector<reco::TransientTrack> & tracks)
  const
{
  if(verbose_) {
    cout << "###################################################" << endl;
    cout << "# DAClusterizerInZT::clusterize   nt="<<tracks.size() << endl;
    cout << "###################################################" << endl;
  }

  vector< vector<reco::TransientTrack> > clusters;
  vector< TransientVertex > pv=vertices(tracks);

  if(verbose_){ cout << "# DAClusterizerInZT::clusterize   pv.size="<<pv.size() << endl;  }
  if (pv.size()==0){ return clusters;}


  // fill into clusters and merge
  vector< reco::TransientTrack>  aCluster=pv.begin()->originalTracks();
  
  if( verbose_ ) { 
      std::cout << '\t' << 0;
      std::cout << ' ' << pv.begin()->position() << ' ' << pv.begin()->time() << std::endl; 
    }

  for(vector<TransientVertex>::iterator k=pv.begin()+1; k!=pv.end(); k++){
    if( verbose_ ) { 
      std::cout << '\t' << std::distance(pv.begin(),k);
      std::cout << ' ' << k->position() << ' ' << k->time() << std::endl; 
    }
    if ( std::abs(k->position().z() - (k-1)->position().z()) > (2*vertexSize_) ||
         std::abs(k->time() - (k-1)->time()) > 2*vertexSizeTime ) {
      // close a cluster
      clusters.push_back(aCluster);
      aCluster.clear();
    }
    for(unsigned int i=0; i<k->originalTracks().size(); i++){ 
      aCluster.push_back( k->originalTracks().at(i)); 
    }
    
  }
  clusters.push_back(aCluster);
  
  if(verbose_) { std::cout << "# DAClusterizerInZT::clusterize clusters.size="<<clusters.size() << std::endl; }
  
  return clusters;

}

