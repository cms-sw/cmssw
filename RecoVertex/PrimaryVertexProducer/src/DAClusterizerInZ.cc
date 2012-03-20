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
}


vector<DAClusterizerInZ::track_t> DAClusterizerInZ::fill(
			  const vector<reco::TransientTrack> & tracks
			  )const{
  // prepare track data for clustering
  vector<track_t> tks;
  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
    track_t t;
    t.z=((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan(((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    //  get the beam-spot
    reco::BeamSpot beamspot=(it->stateAtBeamLine()).beamSpot();
    t.dz2= pow((*it).track().dzError(),2)          // track errror
      + (pow(beamspot.BeamWidthX(),2)+pow(beamspot.BeamWidthY(),2))/pow(tantheta,2)  // beam-width induced
      + pow(vertexSize_,2);                        // intrinsic vertex size, safer for outliers and short lived decays
    if (d0CutOff_>0){
      Measurement1D IP=(*it).stateAtBeamLine().transverseImpactParameter();// error constains beamspot
      t.pi=1./(1.+exp(pow(IP.value()/IP.error(),2)-pow(d0CutOff_ ,2)));  // reduce weight for high ip tracks  
    }else{
      t.pi=1.;
    }
    t.tt=&(*it);
    t.Z=1.;
    tks.push_back(t);
  }
  return tks;
}




// double DAClusterizerInZ::pik(const double beta, const track_t & t, const vertex_t &k )const{
//   //note: t.Z = sum_k exp(-beta*Eik ) is assumed to be valid, this is done in updateAndFit
//   //      then we have sum_k pik(beta, t, k) = 1
//   //      the last call of updateAndFit must have been made with the same temperature ! 
//   //      at low T, the vertex position must be quite accurate (iterated)
//   double Eik=pow(t.z-k.z,2)/t.dz2;
//   double o= k.pk*exp(-beta*Eik); 
//   if(isnan(o)){ cout << "NaN in pik " << k.pk << " " << beta << " " << Eik << " Z=" << t.Z << "   beta Eik=" << beta*Eik << endl;}
//   if (t.Z>0) return k.pk*exp(-beta*Eik)/t.Z; 
//   return 0;
// }


double DAClusterizerInZ::Eik(const track_t & t, const vertex_t &k )const{
  return pow(t.z-k.z,2)/t.dz2;
}
  



double DAClusterizerInZ::update(
					   double beta,
					   vector<track_t> & tks,
					   vector<vertex_t> & y
					   )const{
  //update weights and vertex positions
  // mass constrained annealing without noise
  // returns the squared sum of changes of vertex positions

  //cout << endl << endl << "DAClusterizerInZ::update" << endl;
  unsigned int nt=tks.size();

  //initialize sums
  double sumpi=0;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    k->se=0;    k->sw=0;   k->swz=0;
  }


  // loop over tracks
  for(unsigned int i=0; i<nt; i++){

    // update pik and Zi
    double Zi=0.;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      k->ei=exp(-beta*Eik(tks[i],*k));// cache exponential for one track at a time
      Zi += k->pk * k->ei;
    }
    tks[i].Z=Zi;

    // normalization for pk
    if (tks[i].Z>0){
      sumpi += tks[i].pi;
      // accumulate weighted z and weights for vertex update
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	k->se += tks[i].pi* k->ei / Zi;
	double w = k->pk * tks[i].pi* k->ei / Zi / tks[i].dz2;
	k->sw  += w;
	k->swz += w * tks[i].z;
      }
    }else{
      sumpi += tks[i].pi;
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
      edm::LogInfo("sumw") <<  "invalid sum of weights in fit: " << k->sw << endl;
      if(verbose_){cout << " a cluster melted away ?  pk=" << k->pk <<  " sumw=" << k->sw <<  endl;}
    }

    k->pk = k->pk * k->se / sumpi;
  }

  // return how much the prototypes moved
  return delta;
}





double DAClusterizerInZ::update(
				double beta,
				vector<track_t> & tks,
				vector<vertex_t> & y,
				double & rho0
				)const{
  // MVF style, no more vertex weights, update tracks weights and vertex positions, with noise 
  // returns the squared sum of changes of vertex positions

  unsigned int nt=tks.size();

  //initialize sums
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    k->se=0;    k->sw=0;   k->swz=0;
  }


  // loop over tracks
  for(unsigned int i=0; i<nt; i++){

    // update pik and Zi
    double Zi=rho0*exp(-beta*dzCutOff_*dzCutOff_);// cut-off
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      k->ei=exp(-beta*Eik(tks[i],*k));// cache exponential for one track at a time
      Zi += k->pk * k->ei;
    }
    tks[i].Z=Zi;

    // normalization
    if (tks[i].Z>0){
      //double pi0=exp(-beta*dzCutOff_*dzCutOff_)/tks[i].Z;
      //sumpi += tks[i].pi*(1.-pi0*rho0);  // exclude rho0 from the normalization of the cluster "mass"
       // accumulate weighted z and weights for vertex update
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	k->se += tks[i].pi* k->ei / Zi;
	double w = k->pk * tks[i].pi* k->ei / Zi / tks[i].dz2;
	k->sw  += w;
	k->swz += w * tks[i].z;
      }
    }
    //else{  sumpi += tks[i].pi;  }


  } // end of track loop


  // now update z and pk
  double delta=0;
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( k->sw > 0){
      double znew=k->swz/k->sw; 
      delta+=pow(k->z-znew,2);
      k->z=znew;
    }else{
      edm::LogInfo("sumw") <<  "invalid sum of weights in fit: " << k->sw << endl;
      if(verbose_){cout << " a cluster melted away ?  pk=" << k->pk <<  " sumw=" << k->sw <<  endl;}
    }

    //k->pk *= k->se / sumpi; 
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



bool DAClusterizerInZ::purge(vector<vertex_t> & y, vector<track_t> & tks, double & rho0, const double beta)const{
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
      if(tks[i].Z>0){
	//double p=pik(beta,tks[i],*k);
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
    //rho0+=k0->pk;
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


  double epsilon=1e-3;      // split all single vertices by 10 um 
  double zsep=2*epsilon;    // split vertices that are isolated by at least zsep (vertices that haven't collapsed)
  vector<vertex_t> y1;

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    if ( ( (k==y.begin())|| (k-1)->z < k->z - zsep) && (((k+1)==y.end()  )|| (k+1)->z > k->z + zsep)) { 
      // isolated prototype, split
      vertex_t vnew;
      vnew.z  = k->z -epsilon;
      (*k).z  = k->z+epsilon;
      vnew.pk= 0.5* (*k).pk;
      (*k).pk= 0.5* (*k).pk;
      y1.push_back(vnew);
      y1.push_back(*k);

    }else if( y1.empty() || (y1.back().z < k->z -zsep)){
      y1.push_back(*k);
    }else{
      y1.back().z -=epsilon;
      k->z+=epsilon;
      y1.push_back(*k);
    }
  }// vertex loop
  
  y=y1;
}
 

DAClusterizerInZ::DAClusterizerInZ(const edm::ParameterSet& conf) 
{
  // some defaults to avoid uninitialized variables
  verbose_= conf.getUntrackedParameter<bool>("verbose", false);
  betamax_=0.1;
  betastop_  =1.0;
  coolingFactor_=0.8;
  maxIterations_=100;
  vertexSize_=0.05;  // 0.5 mm
  dzCutOff_=4.0;   // Adaptive Fitter uses 3.0 but that appears to be a bit tight here sometimes

  // configure

  double Tmin = conf.getParameter<double>("Tmin");
  vertexSize_ = conf.getParameter<double>("vertexSize");
  coolingFactor_ = conf.getParameter<double>("coolingFactor");
  d0CutOff_  =  conf.getParameter<double>("d0CutOff");
  dzCutOff_  =  conf.getParameter<double>("dzCutOff");
  maxIterations_=100;
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
  cout << "                                                               z= ";
  cout.precision(4);
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(8) << fixed << k->z ;
  }
  cout << endl << "T=" << setw(15) << 1./beta <<"                                            Tc= ";
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
	  //double p=pik(beta,tks[i],*k);
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
  double rho0=0.0;  // start with no outlier rejection

  vector< TransientVertex > clusters;
  if (tks.empty()) return clusters;

  vector<vertex_t> y; // the vertex prototypes

  // initialize:single vertex at infinite temperature
  vertex_t vstart;
  vstart.z=0.;
  vstart.pk=1.;
  y.push_back(vstart);
  int niter=0;      // number of iterations
  

  // estimate first critical temperature
  double beta=beta0(betamax_, tks, y);
  niter=0; while((update(beta, tks,y)>1.e-6)  && (niter++ < maxIterations_)){ }

  
 // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)
  while(beta<betamax_){ 

    beta=beta/coolingFactor_;
    splitAll(tks,y);

    // make sure we are not too far from equilibrium before cooling further
    niter=0; while((update(beta, tks,y)>1.e-6)  && (niter++ < maxIterations_)){ }

  }


  // merge collapsed clusters 
  while(merge(y,tks.size())){} 
  if(verbose_  ){ cout << "dump after 1st merging " << endl;  dump(beta,y,tks,2);}



  // switch on outlier rejection
  //rho0=exp(beta*dzCutOff_*dzCutOff_)/nt; if(rho0>0.1) rho0=0.1;
  rho0=1./nt; for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ k->pk =1.; }  // democratic
  niter=0; while((update(beta, tks,y,rho0) > 1.e-8)  && (niter++ < maxIterations_)){  }
  if(verbose_  ){ cout << "rho0=" << rho0 <<   " niter=" << niter <<  endl; dump(beta,y,tks,2);}

  
  // merge again  (some cluster split by outliers collapse here)
  while(merge(y,tks.size())){}  
  if(verbose_  ){ cout << "dump after 2nd merging " << endl;  dump(beta,y,tks,2);}


  // continue from freeze-out to Tstop (=1) without splitting, eliminate insignificant vertices
  while(beta<=betastop_){
    while(purge(y,tks,rho0, beta)){
      niter=0; while((update(beta, tks,y,rho0) > 1.e-6)  && (niter++ < maxIterations_)){  }
    } 
    beta/=coolingFactor_;
    niter=0; while((update(beta, tks,y,rho0) > 1.e-6)  && (niter++ < maxIterations_)){  }
  }

  if(verbose_){
   cout << "Final result, rho0=" << rho0 << endl;
   dump(beta,y,tks,2);
  }


  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError; 


  // ensure correct normalization of probabilities, should makes double assginment reasonably impossible
  for(unsigned int i=0; i<nt; i++){  
    tks[i].Z=rho0*exp(-beta*dzCutOff_*dzCutOff_);
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
      tks[i].Z+=k->pk * exp(-beta*Eik(tks[i],*k));
    }
  }


  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
    GlobalPoint pos(0, 0, k->z);
    vector< reco::TransientTrack > vertexTracks;
    for(unsigned int i=0; i<nt; i++){
      if(tks[i].Z>0){
	double p=k->pk * exp(-beta*Eik(tks[i],*k)) / tks[i].Z;
	if( (tks[i].pi>0) && ( p > 0.5 ) ){ vertexTracks.push_back(*(tks[i].tt)); tks[i].Z=0; } // setting Z=0 excludes double assignment
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

