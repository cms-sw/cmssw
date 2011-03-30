#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


using namespace std;


namespace {

  bool recTrackLessZ1(const DAClusterizerInZ::track_t & tk1,
                     const DAClusterizerInZ::track_t & tk2)
  {
    return tk1.z < tk2.z;
  }

  bool vtxLessZ(const DAClusterizerInZ::vertex_t & v1,
                     const DAClusterizerInZ::vertex_t & v2)
  {
    return v1.z < v2.z;
  }

}



vector<DAClusterizerInZ::track_t> DAClusterizerInZ::fill(
			  const vector<reco::TransientTrack> & tracks
			  )const{
  // prepare track data for clustering
  vector<track_t> tks;
  if (tracks.size()==0) { return tks;}

  tks.reserve(tracks.size());
  reco::BeamSpot beamspot=(tracks.begin()->stateAtBeamLine()).beamSpot();

  double sumpi=0;

  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
    track_t t;
    t.z=((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan(((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    //  get the beam-spot
    //reco::BeamSpot beamspot=(it->stateAtBeamLine()).beamSpot();
    t.dz2= pow((*it).track().dzError(),2)          // track errror
      + (pow(beamspot.BeamWidthX(),2)+pow(beamspot.BeamWidthY(),2))/pow(tantheta,2)  // beam-width induced
      + pow(vertexSize_,2);                        // intrinsic vertex size, safer for outliers and short lived decays
    Measurement1D IP=(*it).stateAtBeamLine().transverseImpactParameter();// error contains beamspot
    t.pi=1./(1.+exp(pow(IP.value()/IP.error(),2)-pow(3.,2)));  // reduce weight for high ip tracks , 3 sigma cut-off 
    t.npi=0;
    sumpi+=t.pi;
    t.tt=&(*it);
    t.Z=1.;
    tks.push_back(t);
  }

  // normalize track weights so we don't have to divide by sum(pi) elsewhere
  if(sumpi>0){
    for(vector<track_t>::iterator t=tks.begin(); t!=tks.end(); t++){
      t->npi=t->pi/sumpi;
    }
  }

  return tks;
}





// double DAClusterizerInZ::Eik(const track_t & t, const vertex_t &k )const{
//   return pow(t.z-k.z,2)/t.dz2;
// }
  


double DAClusterizerInZ::update(
					   double beta,
					   vector<track_t> & tks,
					   vector<vertex_t> & y,
					   double Z0
					   )const{
  // update weights and vertex positions
  // Z0=0: mass constrained annealing without noise
  // Z0>0  MVF style, no more vertex weights, update tracks weights and vertex positions, with noise
  //       Z0 = exp(-beta*mu0**2)
  // returns the squared sum of changes of vertex positions

  unsigned int nt=tks.size();

  //initialize sums
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    k->se=0;    k->sw=0;   k->swz=0;
     if(k->pk>0){k->logpk=log(k->pk);}
   }


  // loop over tracks
  for(unsigned int i=0; i<nt; i++){

    track_t & ti=tks.at(i);


    // closest vertex for track i
    double Em=0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      if(k->pk>0){
	k->Ei=beta*Eik(ti,*k) - k->logpk;
	if ((Em==0) || (k->Ei<Em)) { Em=k->Ei;}
      }
    }

    // update pik and Zi
    double Zi=Z0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){

      if((full_)||(beta==0)){
	//k->ei=exp(-beta*k->Ei);
	k->pi=exp(-k->Ei);
	Zi += k->pi;
      }else{
	if(k->Ei < Em + 9.){  // constant term ~ -T*(log(w_min)), log(1e-4)=-9.2
	  k->pi=exp(-k->Ei);
	  Zi += k->pi;
	}else{
	  k->pi=0;
	}
      }

    }
    ti.Z=Zi;


    // normalization for p: sum_k [ k->pk * k->ei] /Zi == 1
    if (Zi>0){
      // accumulate weighted z and weights for vertex update
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	if((k->pi>0)&&(k->pk>0)){
	  double p=ti.npi* k->pi / Zi;
	  double w = p / ti.dz2;
	  k->se  += p/k->pk;
	  k->sw  += w;
	  k->swz += w * ti.z;
	}
      }
    }
    

  } // end of track loop


  // now update z and pk
  double delta=0;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( k->sw > 0){
      double znew=k->swz/k->sw; 
      delta+=pow(k->z-znew,2);
      k->z=znew;
    }else{
      // somehow all tracks fell below the threshold 
      //cout << " invalid sum of weights in update " << scientific << k->sw << "  z=" << k->z << endl;
      edm::LogInfo("sumw") <<  "invalid sum of weights in update: " << k->sw << endl;
    }

    // only update vertex weights before freeze-out (=mass constrained clustering w/o noise)
    if(Z0==0){
      k->pk = k->pk * k->se;
    }

  }


  // return how much the prototypes moved
  return delta;
}



double DAClusterizerInZ::update1(
					   double beta,
					   const bool forceUpdate,
					   vector<track_t> & tks,
					   vector<vertex_t> & y,
					   double Z0
					   )const{
  // update weights and vertex positions
  // Z0=0: mass constrained annealing without noise
  // Z0>0  MVF style, no more vertex weights, update tracks weights and vertex positions, with noise
  //       Z0 = exp(-beta*mu0**2)
  // returns the squared sum of changes of vertex positions

  double delta=0;
  const double zfr=1e-3;
  unsigned int nt=tks.size();

 
  //initialize sums
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      k->se=0;    k->sw=0;   k->swz=0;
      k->update =( (abs(k->z - k->z1 )>zfr ) || ( abs(k->pk - k->pk1)>zfr ));
      if(k->pk>0){k->logpk=log(k->pk);}
  }
  

  // loop over tracks
  for(unsigned int i=0; i<nt; i++){

    track_t & ti=tks[i];

    // closest vertex for track i
    double Em=0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      if(k->pk>0){
	k->Ei=beta*Eik(ti,*k) - k->logpk;
	if ((Em==0) || (k->Ei<Em)) { Em=k->Ei;}
      }
    }

    // update pik and Zi
    double Zi=0;
    if(forceUpdate){
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	if(k->Ei < Em + 9.){  // constant term ~ -T*(log(w_min)), log(1e-4)=-9.2
	  k->pi  =exp(-k->Ei);
	  k->pik[i]=k->pi;
	  Zi += k->pi;
	}else{
	  k->pi=0;
	  k->pik[i]=0;
	}	  
      }
    }else{  // only update when necessary
      Zi=ti.Z;
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
	if ( k->update ){
	  if(k->Ei < Em + 9.){ 
	    double pik=exp(-k->Ei);
	    k->pi=pik;
	    Zi += pik - k->pik[i];
	    k->pik[i]=pik;
	  }else{
	    k->pi=0;
	    Zi=Zi - k->pik[i];
	    k->pik[i]=0;
	  }
	}else{
	  k->pi=k->pik[i];
	}
      }
    }

    
    // normalization for p: sum_k [ k->pk * k->ei] /Zi == 1
    if(Zi>0){
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	if((k->pi>0)&&(k->pk>0)){
	  double p=ti.npi* k->pi / Zi;
	  double w = p / ti.dz2;
	  k->se  += p/k->pk;
	  k->sw  += w;
	  k->swz += w * ti.z;
	}
      }
    }

    ti.Z=Zi;

  } // end of track loop



  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( forceUpdate || k->update ){
      k->z1 =k->z;
      k->pk1=k->pk;
    }
  }


  // now update z and pk
  delta=0;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
 
    if ( k->sw > 0) {
      double znew=k->swz/k->sw; 
      delta+=pow(k->z-znew,2);
      k->z=znew;
    }else{
      // somehow all tracks fell below the threshold 
      cout << " invalid sum of weights in update1 " << scientific << k->sw << "  z=" << k->z << endl;
      cout << k->z << " "  << k->z1 << " " <<  k->pk << " " << k->se << endl;
      edm::LogInfo("sumw") <<  "invalid sum of weights in update: " << k->sw << endl;
      k->pk=0;  // eliminate it in the next splitAll
    }
    // only update vertex weights before freeze-out (=mass constrained clustering w/o noise)
    if(Z0==0){
      k->pk = k->pk * k->se;
    }
  }

  

  // return how much the prototypes moved
  return delta;
}









bool DAClusterizerInZ::merge(vector<vertex_t> & y, int nt)const{
  // merge clusters that collapsed or never separated, return true if vertices were merged, false otherwise

 if(y.size()<2)  return false;

  for(vector<vertex_t>::iterator k=y.begin(); (k+1)!=y.end(); k++){
    //if ((k+1)->z - k->z<1.e-2){  // note, no fabs here, maintains z-ordering  (with split()+merge() at every temperature)
    if (fabs((k+1)->z - k->z)<1.e-2){  // with fabs if only called after freeze-out (splitAll() at highter T)
      k->pk += (k+1)->pk;
      k->z=0.5*(k->z+(k+1)->z);
      y.erase(k+1);
      return true;  
    }
  }
  
  return false;
}



bool DAClusterizerInZ::purge(vector<vertex_t> & y, vector<track_t> & tks, double & Z0, const double beta)const{
  // eliminate clusters with only one significant/unique track
  if(y.size()<2)  return false;
  
  unsigned int nt=tks.size();
  double sumpmin=nt;
  vector<vertex_t>::iterator k0=y.end();
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
    int nUnique=0;
    double sump=0;
    double pmax=k->pk/(k->pk+Z0);
    for(unsigned int i=0; i<nt; i++){
      if(tks[i].Z>0){
	double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z;
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
    if(verbose_){cout << "eliminating prototype at " << k0->z << " with sump=" << sumpmin << endl;}
    y.erase(k0);
    return true;
  }else{
    return false;
  }
}


 

double DAClusterizerInZ::beta0(
			       double betamax,
			       vector<track_t> & tks,
			       vector<vertex_t> & y
			       )const{
  
  double T0=0;  // max Tc for beta=0
  // estimate critical temperature from beta=0 (T=inf)
  unsigned int nt=tks.size();

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){

    // vertex fit at T=inf 
    double sumwz=0;
    double sumw=0;
    for(unsigned int i=0; i<nt; i++){
      double w=tks[i].pi/tks[i].dz2;
      sumwz+=w*tks[i].z;
      sumw +=w;
    }
    k->z=sumwz/sumw;

    // estimate Tcrit, eventually do this in the same loop
    double a=0, b=0;
    for(unsigned int i=0; i<nt; i++){
      double dx=tks[i].z-(k->z);
      double w=tks[i].pi/tks[i].dz2;
      a+=w*pow(dx,2)/tks[i].dz2;
      b+=w;
    }
    double Tc= 2.*a/b;  // the critical temperature of this vertex
    if(Tc>T0) T0=Tc;
  }// vertex loop (normally there should be only one vertex at beta=0)
  
  if (T0>1./betamax){
    return betamax/pow(coolingFactor_, int(log(T0*betamax)/log(coolingFactor_))-1 );
  }else{
    // ensure at least one annealing step
    return betamax/coolingFactor_;
  }
}



void DAClusterizerInZ::splitAll(
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{

  double epsilon=1e-3;
  if(vertexSize_*0.25>epsilon){
    epsilon=vertexSize_*0.25;
  }
  double zsep=2*epsilon;    // split vertices that are isolated by at least zsep (vertices that haven't collapsed)
  vector<vertex_t> y1;
  y1.reserve(2*y.size());

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if(k->pk>0){
    if ( ( (k==y.begin())|| (k-1)->z < k->z - zsep) && (((k+1)==y.end()  )|| (k+1)->z > k->z + zsep)) { 
      // isolated prototype, split
      vertex_t vnew;
      vnew.z  = k->z - epsilon;
      (*k).z  = k->z + epsilon;
      vnew.pk= 0.5* (*k).pk;
      (*k).pk= 0.5* (*k).pk;
      vnew.pik.reserve(k->pik.size());
      vnew.pik=k->pik;
      y1.push_back(vnew);
      y1.push_back(*k);

    }else if( y1.empty() || (y1.back().z < k->z -zsep)){
      y1.push_back(*k);
    }else{
      y1.back().z -=epsilon;
      k->z+=epsilon;
      y1.push_back(*k);
    }
    }else{
    }
  }// vertex loop
  
  y=y1;
}
 

DAClusterizerInZ::DAClusterizerInZ(const edm::ParameterSet& conf) 
{
  // some defaults to avoid uninitialized variables
  verbose_= conf.getUntrackedParameter<bool>("verbose", false);
  //splitMergedClusters_= conf.getUntrackedParameter<bool>("splitMergedClusters", false);
  //mergeAfterAnnealing_= conf.getUntrackedParameter<bool>("mergeAfterAnnealing", true);
  //useTrackResolutionAfterFreezeOut_= conf.getUntrackedParameter<bool>("useTrackResolutionAfterFreezeOut", false);
  //full_=conf.getUntrackedParameter<bool>("full",true);
  //deltamax_=  conf.getUntrackedParameter<double>("deltamax",1.e-6);
  splitMergedClusters_= false;
  mergeAfterAnnealing_= true;
  useTrackResolutionAfterFreezeOut_= false;
  full_=false;
  deltamax_=  1.e-6;
  betamax_=0.1;
  betastop_  =1.0;
  coolingFactor_=0.8;
  maxIterations_=100;
  vertexSize_=0.01;  // 0.1 mm
  mu0_=4.0;   // Adaptive Fitter uses 3.0 but that appears to be a bit tight here sometimes

  // configure

  double Tmin = conf.getParameter<double>("Tmin");
  vertexSize_ = conf.getParameter<double>("vertexSize");
  coolingFactor_ = conf.getParameter<double>("coolingFactor");

  if (Tmin==0){
    cout << "DAClusterizerInZ: invalid Tmin" << Tmin << "  reset do default " << 1./betamax_ << endl;
  }else{
    betamax_ = 1./Tmin;
  }

}


void DAClusterizerInZ::dump(const double beta, const vector<vertex_t> & y, const vector<track_t> & tks0, int verbosity)const{

  // copy and sort for nicer printout
  vector<track_t> tks; 
  for(vector<track_t>::const_iterator t=tks0.begin(); t!=tks0.end(); t++){tks.push_back(*t); }
  stable_sort(tks.begin(), tks.end(), recTrackLessZ1);

  cout << "-----DAClusterizerInZ::dump ----" << endl;
  cout << "beta=" << beta << "   betamax= " << betamax_ << endl;
  cout << "                                                                z= ";
  cout.precision(4);
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(8) << fixed << k->z ;
  }
  cout << endl << "T=" << setw(15) << 1./beta <<"                                              Tc=";
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    double a=0, b=0;
    for(unsigned int i=0; i<tks.size(); i++){
      if((tks[i].pi>0)&&(tks[i].Z>0)){
	double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z; 
	double dx=tks[i].z-(k->z);
	double w=p*tks[i].pi/tks[i].dz2;
	a+=w*pow(dx,2)/tks[i].dz2;
	b+=w;
      }
    }
    if(b>0){
      double Tc= 2.*a/b;
      cout <<  setw(8) <<  setprecision(3) <<  fixed << Tc;
    }
  }
  cout << endl << "                                                               pk=";
  double sumpk=0;
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout <<  setw(8) <<  setprecision(3) <<  fixed << k->pk;
    sumpk+=k->pk;
  }
  cout  << endl;

  if(verbosity>0){
    double E=0, F=0;
    cout << endl;
    cout << "----       z +/- dz                ip +/-dip       pt    phi  eta    weights  ----" << endl;
    cout.precision(4);
    for(unsigned int i=0; i<tks.size(); i++){
      if (tks[i].Z>0){	F-=log(tks[i].Z)/beta;}
      double tz= tks[i].z;
      cout <<  setw (3)<< i << ")" <<  setw (8) << fixed << setprecision(4)<<  tz << " +/-" <<  setw (6)<< sqrt(tks[i].dz2);

      if(tks[i].tt->track().quality(reco::TrackBase::highPurity)){ cout << " *";}else{cout <<"  ";}
      if(tks[i].tt->track().hitPattern().hasValidHitInFirstPixelBarrel()){cout <<"+";}else{cout << "-";}
      cout << setw(1) << tks[i].tt->track().hitPattern().pixelBarrelLayersWithMeasurement(); // see DataFormats/TrackReco/interface/HitPattern.h
      cout << setw(1) << tks[i].tt->track().hitPattern().pixelEndcapLayersWithMeasurement(); 
      cout << setw(1) << hex << tks[i].tt->track().hitPattern().trackerLayersWithMeasurement()-tks[i].tt->track().hitPattern().pixelLayersWithMeasurement() <<dec; 
      cout << "=" << setw(1)<<hex <<tks[i].tt->track().trackerExpectedHitsOuter().numberOfHits() << dec;

      Measurement1D IP=tks[i].tt->stateAtBeamLine().transverseImpactParameter();
      cout << setw (8) << IP.value() << "+/-" << setw (6) << IP.error();
      cout << " " << setw(6) << setprecision(2)  << tks[i].tt->track().pt()*tks[i].tt->track().charge();
      cout << " " << setw(5) << setprecision(2) << tks[i].tt->track().phi() 
	   << " "  << setw(5)  << setprecision(2)   << tks[i].tt->track().eta() ;

      double sump=0.;
      for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
	if((tks[i].pi>0)&&(tks[i].Z>0)){
	  double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z; 
	  if( p > 0.0001){
	    cout <<  setw (8) <<  setprecision(3) << p;
	  }else{
	    cout << "    .   ";
	  }
	  E+=p*Eik(tks[i],*k);
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
DAClusterizerInZ::vertices(const vector<reco::TransientTrack> & tracks, const int verbosity) 
const
{

 
  vector<track_t> tks=fill(tracks);
  unsigned int nt=tracks.size();
  double Z0=0.0;  // start with no outlier rejection


  vector< TransientVertex > clusters;
  if (tks.empty()) return clusters;

  vector<vertex_t> y; // the vertex prototypes

  // initialize:single vertex at infinite temperature
  vertex_t vstart;
  vstart.z=0.;
  vstart.pk=1.;
  vstart.pik.reserve(tks.size());
  vstart.pik.assign(tks.size(), 1.);
  y.push_back(vstart);
  int niter=0;      // number of iterations
  

  // estimate the first critical temperature
  double beta=beta0(betamax_, tks, y);
  niter=0; while((update(beta, tks,y)>deltamax_)  && (niter++ < maxIterations_)){ }

 // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)
  while(beta<betamax_){ 

    beta=beta/coolingFactor_;
    std::sort(y.begin(), y.end(), vtxLessZ);
    splitAll(tks,y);

    // make sure we are not too far from equilibrium before cooling further
    if (full_){
      niter=0; while((update(beta, tks,y)>deltamax_)  && (niter++ < maxIterations_)){ }
    }else{
      //niter=0; while((update(beta/pow(coolingFactor_,double(niter)/double(maxIterations_)), tks,y)>deltamax_)  && (niter++ < maxIterations_)){ }
      niter=0;
      double delta=2*deltamax_;
      while((delta>deltamax_)&&(niter++ < maxIterations_)){
	delta=update1(beta, (niter==1), tks, y);
      }
    }

  }


  // merge collapsed clusters 
  while(merge(y,tks.size())){} 
  if(verbose_  ){ cout << "dump after freeze-out and 1st merging " << endl;  dump(beta,y,tks,2);}



  // switch on outlier rejection

  if(useTrackResolutionAfterFreezeOut_){
    for(unsigned int i=0; i<nt; i++){ tks[i].dz2-=pow(vertexSize_,2);} // use the real resolution from here on
  }

  //double rho0=1./nt; 
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ k->pk =1.; }  // democratic
  //Z0=rho0*exp(-beta*mu0_*mu0_);
  Z0=exp(-beta*mu0_*mu0_);
  niter=0; while((update(beta, tks,y, Z0) > deltamax_)  && (niter++ < maxIterations_)){  }
  if(verbose_  ){ cout <<  " niter=" << niter <<  endl; dump(beta,y,tks,2);}

  
  // merge again  (some clusters that were split by outliers collapse here)
  while(merge(y,tks.size())){}  
  if(verbose_  ){ cout << "dump after outlier rejection and 2nd merging " << endl;  dump(beta,y,tks,2);}


  if(splitMergedClusters_){
    // create new seeds from rejected tracks
    int nseed=0;
    for(unsigned int i=0; i<nt; i++){
      double sump=0;
      for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
	if((tks[i].pi>0)&&(tks[i].Z>0)){
	  double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z; 
	  sump+=p;
	}
      }
      if((sump<0.1)&&(tks[i].pi>0.5)){
	nseed++;
	vertex_t vnew;
	vnew.z = tks[i].z;
	vnew.pk =1;
	vector<vertex_t>::iterator k=y.begin();
	for( ; (k!=y.end()) && (k->z<vnew.z); k++){}
	y.insert(k, vnew);
	if(verbose_){cout << "new vertex seeded by downweighted track " <<  setw (8) << fixed << setprecision(4) << vnew.z << endl;  }
      }
    }
    
    if (nseed>0){
      niter=0; while((update(beta, tks,y,Z0) > deltamax_)  && (niter++ < maxIterations_)){  }
      if (verbose_){     cout << "dump after outlier seeding" << endl;     dump(beta,y,tks,2);}
    }else{
      if (verbose_){cout << "no new seeds after freezeout" << endl;}
    }
  }//


  // continue from freeze-out to Tstop (=1) without splitting, eliminate insignificant vertices
  while(beta<=betastop_){
    while(purge(y,tks,Z0, beta)){
      niter=0; while((update(beta, tks,y,Z0) > deltamax_)  && (niter++ < maxIterations_)){  }
    } 
    beta/=coolingFactor_;
    //Z0=rho0*exp(-beta*mu0_*mu0_);
    Z0=exp(-beta*mu0_*mu0_);
    niter=0; while((update(beta, tks,y,Z0) > deltamax_)  && (niter++ < maxIterations_)){  }
  }


  if(verbose_){
   dump(beta,y,tks,2);
  }


  // ensure correct normalization of probabilities, should makes double assginment reasonably impossible
  for(unsigned int i=0; i<nt; i++){
    tks[i].Z=0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      tks[i].Z+=k->pk * exp(-beta*Eik(tks[i],*k));
    }
  }



  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    GlobalPoint pos(0, 0, k->z);
    vector< reco::TransientTrack > vertexTracks;
    for(unsigned int i=0; i<nt; i++){
      if(tks[i].Z>0){
        double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z;
        if( (tks[i].pi>0) && ( p > 0.5 ) ){ 
	  vertexTracks.push_back(*(tks[i].tt)); 
	  tks[i].Z=0;  // makes double assginment impossible
	}
      }
    }
    TransientVertex v(pos, dummyError, vertexTracks, 0);
    clusters.push_back(v);
  }


  return clusters;

}





vector< vector<reco::TransientTrack> >
DAClusterizerInZ::clusterize(const vector<reco::TransientTrack> & tracks)
  const
{
  if(verbose_) {
    cout << "###################################################" << endl;
    cout << "# DAClusterizerInZ::clusterize   nt="<<tracks.size() << endl;
    cout << "###################################################" << endl;
  }

  vector< vector<reco::TransientTrack> > clusters;
  vector< TransientVertex > pv=vertices(tracks);

  if(verbose_){ cout << "# DAClusterizerInZ::clusterize   pv.size="<<pv.size() << endl;  }
  if (pv.size()==0){ return clusters;}

  if(!mergeAfterAnnealing_){
    // fill into clusters, no merging
    for(vector<TransientVertex>::iterator k=pv.begin(); k!=pv.end(); k++){
      clusters.push_back(k->originalTracks());
    }
    return clusters;
  }

  // fill into clusters and merge
  vector< reco::TransientTrack>  aCluster=pv.begin()->originalTracks();
  
  for(vector<TransientVertex>::iterator k=pv.begin()+1; k!=pv.end(); k++){
    if ( k->position().z() - (k-1)->position().z()> (2*vertexSize_) ){
      // close a cluster
      clusters.push_back(aCluster);
      aCluster.clear();
    }
    for(unsigned int i=0; i<k->originalTracks().size(); i++){ aCluster.push_back( k->originalTracks().at(i)); }
    
  }
  clusters.push_back(aCluster);
  
  
  return clusters;

}

