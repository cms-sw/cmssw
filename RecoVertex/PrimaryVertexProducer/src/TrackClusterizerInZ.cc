#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"


using namespace std;


namespace {

  bool recTrackLessZ1(const TrackClusterizerInZ::track_t & tk1,
                     const TrackClusterizerInZ::track_t & tk2)
  {
    return tk1.z < tk2.z;
  }


  bool recTrackLessZ(const reco::TransientTrack & tk1,
                     const reco::TransientTrack & tk2)
  {
    return tk1.stateAtBeamLine().trackStateAtPCA().position().z() < tk2.stateAtBeamLine().trackStateAtPCA().position().z();
  }

}


vector<TrackClusterizerInZ::track_t> TrackClusterizerInZ::fill(
			  const vector<reco::TransientTrack> & tracks
			  )const{
  // prepare track data for clustering
  vector<track_t> tks;
  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
    track_t t;
    t.z=((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan(((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    Measurement1D IP=(*it).stateAtBeamLine().transverseImpactParameter();
    t.ip =IP.value();
    t.dip=IP.error();
    t.pt=it->track().pt();
    //  get the beam-spot
    reco::BeamSpot beamspot=(it->stateAtBeamLine()).beamSpot();
    double x0,y0,wxy0;// can't make these members because of const (why is everything const ?)
    x0=beamspot.x0();
    y0=beamspot.y0();
    wxy0=beamspot.BeamWidthX();
    t.dz2= pow((*it).track().dzError(),2)+pow(wxy0/tantheta,2);

    t.tt=&(*it);
    t.clu=-1;
    tks.push_back(t);
  }
  return tks;
}



void TrackClusterizerInZ::updateWeights(
			 double beta,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{

  unsigned int nt=tks.size();
  double delta=0;
  // evaluate assignment probabilities p(track,vertex), Z(track) and p(vertex)
  for(unsigned int i=0; i<nt; i++){

    tks[i].Z=0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      double dx=tks[i].z-(*k).z;                      
      k->ptk[i] = k->py*exp(-beta*dx*dx/tks[i].dz2); 
      if (DEBUG && isnan(k->ptk[i])){
	cout << "nan in ptk " << k->py << "," << beta << ",tks[i].z" << tks[i].z<< ",k.z=" << k->z  << "," << tks[i].dz2 << "," << i << endl;
      }
      tks[i].Z += k->ptk[i];                           // Z_i = sum_k p_k exp(-beta d_ik^2)
    }

    // normalize the p_ik:  p_ik= p_k exp(-beta d_ik^2) / sum_k' p_k' exp(-beta d_ik'^2) = p_k exp(-beta d_ik^2 )/Z_i
    if (tks[i].Z>0){
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	k->ptk[i]/=tks[i].Z;
      }
    }else{
      // a track dropped out
      if (verbose_){cout <<"TrackClusterizerInZ::updateWeights  Zi=" << tks[i].Z << "  !!!!!!!!!!!!!!    i=" << i<< endl;}
    }
  }

  // update vertex weights
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    double sump=0;
    for(unsigned int i=0; i<nt; i++){
      sump += k->ptk[i];
    }
    delta+=pow(k->py-sump/nt,2);
    k->py=sump/nt;
  }

  // update critical temperatures  (not really needed here, but nice for debugging)
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    double a=0, b=0;
    for(unsigned int i=0; i<nt; i++){
      double dxi2=pow(tks[i].z-(k->z),2)/tks[i].dz2;
      double sig2=tks[i].dz2;
      double pik=k->ptk[i];
      a+=dxi2/sig2*pik;
      b+=pik/sig2;
    }
    k->Tc=2.*a/b;
  }

}





double TrackClusterizerInZ::fit(double beta,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{
  /* do a single fit at fixed temperature, assume that weights are up to date
     returns the squared sum of changes of vertex positions */
  
  // "fit", get the weighted mean of track z for each vertex candidate
  double delta=0;
  unsigned int nt=tks.size();

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    
    double sumwx=0, sumwy=0, sumwz=0;
    double sumw=0;
    for(unsigned int i=0; i<nt; i++){
      double w=k->ptk[i]/tks[i].dz2;  // weight * 1/sigma_z^2
      sumwx+=w*tks[i].x;
      sumwy+=w*tks[i].y;
      sumwz+=w*tks[i].z;
      sumw +=w;
    }
    if(sumw>0){
      double y1=sumwz/sumw; 
      delta+=pow(k->z-y1,2);
      k->z=y1;
      k->x=sumwx/sumw;
      k->y=sumwy/sumw;
      if (isnan(y1)) {cout << "nan in fit: " << sumwz << "/" << sumw << endl;}
    }else{
      cout << "sumw="<< sumw << " in TrackClusterizerInZ::fit" << endl; 
    }
    
  }// vertex loop

  return delta;
}



bool TrackClusterizerInZ::merge(vector<vertex_t> & y, int nt)const{
  // merge clusters that collapsed, return true if vertices were merged, false otherwise
  if(y.size()<2)  return false;
  for(vector<vertex_t>::iterator k=y.begin(); (k+1)!=y.end(); k++){
    if ((k+1)->z - k->z<1.e-4){  // note, no fabs here, maintains z-ordering
      k->py += (k+1)->py;
      for(int i=0; i<nt; i++){
	k->ptk[i]+=(k+1)->ptk[i];
      }
      delete (k+1)->ptk;
      y.erase(k+1);
      return true;  
    }
  }
  return false;
}








 

double TrackClusterizerInZ::beta0(
				  double betamax,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{
  
  double T0=0;  // max Tc for beta=0

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    double a=0, b=0;
    for(unsigned int i=0; i<tks.size(); i++){
      double dxi2=pow(tks[i].z-(k->z),2)/tks[i].dz2;
      double sig2=tks[i].dz2;
      double pik=k->ptk[i];
      a+=dxi2/sig2*pik;
      b+=pik/sig2;
    }
    k->Tc=2.*a/b;  // the critical temperature of this vertex
    if (k->Tc>T0) {T0=k->Tc;}
  }// vertex loop
  
  return betamax/pow(coolingFactor_, int(log(T0*betamax)/log(coolingFactor_))-1 );
}



double TrackClusterizerInZ::split(
			 double beta,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{
  
  unsigned int nt=tks.size();
  vector<vertex_t> y1;
  double T0=0;  // max Tc for beta=0
  int nsplit=0;

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    
    double a=0, b=0;
    double Ak=0, Bk=0;
    //double Ck=0;
    for(unsigned int i=0; i<nt; i++){
      double dxi2=pow(tks[i].z-(k->z),2)/tks[i].dz2;
      double sig2=tks[i].dz2;
      double pik=k->ptk[i];
      a+=dxi2/sig2*pik;
      b+=pik/sig2;
      double u  =dxi2*beta;
      double aik=pik*(2.*u-1.)/sig2;
      //double bik=pik*(4.*u*u-12.*u+3.)/6./sig2/sig2*beta;
      Ak+=-aik;
      //Bk+=-bik+0.5*beta*aik*aik;
      Bk+=beta*pik/sig2/sig2*( u*u*(2*pik-2./3.) - 0.5*(1.-pik)*(1.-4*u) );
    }
    k->Tc=2.*a/b;  // the critical temperature of this vertex

    if (beta==0){      // no splitting at infinite Temperature, just remember Tc

      if (k->Tc>T0) {T0=k->Tc;}

    }else{  // beta>0, finite Temperature
      //      if(true){	k->epsilon=1e-3;// just a test
      if( (k->Tc > (1/beta)) && (Ak<0)  && (Bk>0) ){
	k->epsilon=sqrt(-0.5*Ak/Bk);  // preferred split distance 
       if(verbose_){std::cout << "splitting vertex at " << (*k).z << "   by +/-" << (*k).epsilon  <<std::endl;}
       nsplit++;
       vertex_t vnew;
       vnew.z  =(*k).z-(*k).epsilon;
       (*k).z  =(*k).z+(*k).epsilon;
       vnew.py= 0.5* (*k).py;
       (*k).py= 0.5* (*k).py;
       vnew.x=(*k).x;
       vnew.y=(*k).y;
       vnew.ptk = new double[nt];
       for(unsigned int i=0; i<nt; i++){ vnew.ptk[i]=(*k).ptk[i]*0.5; (*k).ptk[i]=(*k).ptk[i]*0.5;}
       vnew.Tc=0;
       y1.push_back(vnew);
       y1.push_back(*k);

      }else{
	y1.push_back(*k);
      }
    }
  }// vertex loop
  

  if(beta==0){
    // don't split, just set the start temperature
    return betamax_/pow(coolingFactor_, int(log(T0*betamax_)/log(coolingFactor_))-1 );
  }else{
    y=y1;
    return beta/coolingFactor_;
  }
}
 

TrackClusterizerInZ::TrackClusterizerInZ(const edm::ParameterSet& conf) 
{
  // some defaults to avoid uninitialized variables
  verbose_= conf.getUntrackedParameter<bool>("verbose", false);
  DEBUG   =false;
  betamax_=0.1;
  coolingFactor_=0.8;
  maxIterations_=100;

  // configure
  algorithm = conf.getParameter<std::string>("algorithm");
  if (algorithm=="gap"){
    zSep = conf.getParameter<edm::ParameterSet>("TkGapClusParameters").getParameter<double>("zSeparation");
  }else if(algorithm=="DA"){
    double Tmin = conf.getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("Tmin");
    zSep = conf.getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("zSeparation");
    coolingFactor_ = conf.getParameter<edm::ParameterSet>("TkDAClusParameters").getParameter<double>("coolingFactor");
    maxIterations_=100;
    if (Tmin==0){
      cout << "TrackClusterizerInZ: invalid Tmin" << Tmin << "  reset do default " << 1./betamax_ << endl;
    }else{
      betamax_ = 1./Tmin;
    }
  }else{
    throw VertexException("TrackClusterizerInZ: unknown algorithm: " + algorithm);
  }
}


TrackClusterizerInZ::TrackClusterizerInZ(){
  zSep = 0.1;
  betamax_=0.1;
  maxIterations_=100;
  verbose_=false;
  DEBUG=false;
  coolingFactor_=0.8;
}



void TrackClusterizerInZ::dump(const double beta, const vector<vertex_t> & y, const vector<track_t> & tks, int verbosity)const{
  cout << "-----TrackClusterizerInZ::dump ----" << endl;
  cout << "beta=" << beta << "   betamax= " << betamax_ << endl;
  cout << "                                                  z= ";
  cout.precision(4);
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(10) << fixed << k->z ;
  }
  cout << endl << "T=" << setw(15) << 1./beta <<"                                Tc= ";
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout <<  setw(10) <<  fixed << k->Tc;
  }
  cout << endl << "pk=                                                  ";
  double sumpk=0;
  for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
    cout <<  setw(10) <<  fixed << k->py;
    sumpk+=k->py;
  }
  cout  << endl;

  if(verbosity>0){
    cout << endl;
    cout << "----       z +/- dz            ip +/-dip          pt     weights  ----" << endl;
    cout.precision(4);
    for(unsigned int i=0; i<tks.size(); i++){
      cout <<  setw (3)<< i << ")" <<  setw (8) << fixed << tks[i].z << " +/-" <<  setw (6)<< sqrt(tks[i].dz2);
      cout << "   " << setw (8) << tks[i].ip << " +/-" << setw (6) << tks[i].dip;
      cout << "  " << setw(8) << tks[i].pt;

      double sump=0.;
      for(vector<vertex_t>::const_iterator k=y.begin(); k!=y.end(); k++){
	cout <<  " " << setw (9) << k->ptk[i];
	sump+=k->ptk[i];
      }
      cout << endl;
    }
  }
  cout << endl << "----------" << endl;
}




vector< TransientVertex >
TrackClusterizerInZ::vertices(const vector<reco::TransientTrack> & tracks, const double Tmin) 
const
{

  // allow caller to override configuration
  double betamax=betamax_;
  if(Tmin>0){  betamax=1./Tmin;  }

  vector<track_t> tks=fill(tracks);
  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), recTrackLessZ1);
  unsigned int nt=tracks.size();

  vector< TransientVertex > clusters;
  if (tks.empty()) return clusters;


  vector<vertex_t> y; // the vertex prototypes

  // initialize:single vertex at infinite temperature
  vertex_t vstart;
  vstart.z=0.;
  vstart.x=0.;
  vstart.y=0.;
  vstart.py=1;
  vstart.ptk = new double[nt];  // delete it afterwards !
  vstart.Tc=0;
  for(unsigned int i=0; i<nt; i++){
    vstart.ptk[i]=1.;
  }
  y.push_back(vstart);
  

  double delta=fit(0, tks, y);
  if(verbose_){cout << "first fit: " << endl; dump(0,y,tks,1);}

  //  double beta=split(0,tks,y);
  double beta=beta0(betamax, tks, y);

  while(beta<betamax){  // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)

    split(beta, tks, y);   // reduce temperature or split vertices
    updateWeights(beta, tks,y);

    if(verbose_) {cout << "after split " << endl; dump(beta,y,tks,1);}

    // find equilibrium at new T
    delta=1.;
    int niter=0;      // number of iterations
    while((delta>1.e-8)  && (niter++ < maxIterations_)){ 

      delta=fit(beta,tks, y);
      updateWeights(beta, tks,y);
      niter++;

    }

    if (verbose_) {cout << "after fit   niter=" << niter << "  delta=" << delta << endl; dump(beta,y,tks,1);}
    while(merge(y,tks.size())){}

    beta=beta/coolingFactor_;

  } // annealing loop, stop when T<Tmin



  //one more at the freeze-out temperatue
  for(int i=0; i<3; i++){
    split(betamax, tks, y);   // reduce temperature or split vertices
    delta=1.;
    int niter=0;   
    while((delta>1.e-8)  && (niter++ < maxIterations_)){ 

      delta=fit(betamax,tks, y);
      updateWeights(betamax, tks,y);
      niter++;

    }
    while(merge(y,tks.size())){}
  }
  if(verbose_){
    cout << "freezout" << endl;
    dump(beta,y,tks,1);
  }

  // freeze-out, stop splitting but cool down further to get the assignment
  while(beta<10.){
      updateWeights(beta, tks,y);
      delta=fit(beta,tks, y);
      beta/=coolingFactor_;
  }

  // end of annealing
  if(verbose_){ dump(beta,y,tks,2); }

  // select significant tracks and use a TransientVertex as a container
  GlobalError dummyError; 
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ 
    GlobalPoint pos(k->x, k->y, k->z);
    vector< reco::TransientTrack > vertexTracks;
    for(unsigned int i=0; i<nt; i++){
      if((k->ptk[i])>0.5){ vertexTracks.push_back(*(tks[i].tt)); }
    }
    TransientVertex v(pos, dummyError, vertexTracks, 0);

    clusters.push_back(v);
    delete k->ptk;   // avoid a memory leak
  }

  if(verbose_) {cout << "clusterizeDA returns " << clusters.size() << " clusters " << endl;}
  return clusters;

}







float TrackClusterizerInZ::zSeparation() const 
{
  return zSep;
}




vector< vector<reco::TransientTrack> >
TrackClusterizerInZ::clusterize0(const vector<reco::TransientTrack> & tracks)
  const
{
  
  vector<reco::TransientTrack> tks = tracks; // copy to be sorted

  vector< vector<reco::TransientTrack> > clusters;
  if (tks.empty()) return clusters;

  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), recTrackLessZ);

  // init first cluster
  vector<reco::TransientTrack>::const_iterator it = tks.begin();
  vector <reco::TransientTrack> currentCluster; currentCluster.push_back(*it);

  it++;
  for ( ; it != tks.end(); it++) {
    double zPrev = currentCluster.back().stateAtBeamLine().trackStateAtPCA().position().z();
    double zCurr = (*it).stateAtBeamLine().trackStateAtPCA().position().z();

    if ( abs(zCurr - zPrev) < zSeparation() ) {
      // close enough ? cluster together
      currentCluster.push_back(*it);
    }
    else {
      // store current cluster, start new one
      clusters.push_back(currentCluster);
      currentCluster.clear();
      currentCluster.push_back(*it);
      // it++; if (it == tks.end()) break;
    }
  }

  // store last cluster
  clusters.push_back(currentCluster);

  return clusters;

}



vector< vector<reco::TransientTrack> >
TrackClusterizerInZ::clusterize1(const vector<reco::TransientTrack> & tracks)
  const
{
  // modified gap clustering: 
  //  use only tracks with a zresolution better than the 0.5 zSep for the gap search
  //  then distribute remaining tracks among clusters
  vector<track_t> tks=fill(tracks);
  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), recTrackLessZ1);

  vector< vector<reco::TransientTrack> > clusters;
  if (tks.empty()) return clusters;

  vector< vector<track_t> > tclusters;
  // init first cluster
  vector<track_t>::iterator it = tks.begin();
  vector <track_t> currentCluster; 
  double zSep  =zSeparation()-1.0;
  double zscale=0.5*(zSeparation()-1.0);
  

  it++;
  for ( ; it != tks.end(); it++) {
    
    if(it->dz2>zscale) continue;

    if(currentCluster.empty()){
      it->clu=tclusters.size(); currentCluster.push_back(*it);
    }else{
      double zPrev = currentCluster.back().z;
      double zCurr = (*it).z;
      if ( abs(zCurr - zPrev) < zSep ) {
	// close enough ? cluster together
	it->clu=tclusters.size(); currentCluster.push_back(*it);
      } else {
	// store current cluster, start new one
	tclusters.push_back(currentCluster);
	currentCluster.clear();
	it->clu=tclusters.size(); currentCluster.push_back(*it);
	it++; if (it == tks.end()) break;
      }
    }
  }
  
  // store last cluster
  tclusters.push_back(currentCluster);


  // take care of low resolution tracks
  zscale*=sqrt(2);
  while (zscale<=1.0){
    // prepare cluster info
    unsigned int nclu=tclusters.size();
    vector<double> zmin(nclu,99.), zmax(nclu,-99.),zmean(nclu,0.);
    for(unsigned int i=0; i<nclu; i++){
      double w=0, wz=0;
      for(vector<track_t>::iterator it=tclusters[i].begin(); it!=tclusters[i].end(); it++){
	if (it->z>zmax[i]) zmax[i]=it->z;
	if (it->z<zmin[i]) zmin[i]=it->z;
	w +=1./it->dz2;
	wz+=it->z/it->dz2;
      }
      zmean[i]=wz/w;
    }
    
    // now throw in tracks that are inside a cluster or less than 2 sigma away from the mean
    for(it=tks.begin(); it!=tks.end(); it++){
      if(it->clu>=0) continue;  // already assigned
      int clu=-1;
      for(unsigned int i=0; i<nclu; i++){
	if((it->z>zmin[i])&&(it->z<zmax[i])){ clu=i;}
	else{
	  if ( (sqrt(it->dz2)<zscale) && (fabs(it->z-zmean[i])<(2*sqrt(it->dz2))) ){
	    if( (clu<0)||( (clu>=0) && (fabs(it->z-zmean[i])<fabs(it->z-zmean[clu]))) ) clu=i;
	  }
	}
      }
      if(clu>=0){  it->clu=clu; tclusters[clu].push_back(*it);}
    }
    zscale*=sqrt(2);
  }

  // finally return vector < vector TransientTrack >> 
  vector<vector<reco::TransientTrack > > result;
  for(vector< vector<track_t> >::iterator ic=tclusters.begin(); ic!=tclusters.end(); ic++){
    vector <reco::TransientTrack> aCluster; 
    for(vector<track_t>::iterator it=ic->begin(); it!=ic->end(); it++){
      aCluster.push_back(*(it->tt));
    }
    result.push_back(aCluster);
  }
  return result;
  }




vector< vector<reco::TransientTrack> >
TrackClusterizerInZ::clusterize(const vector<reco::TransientTrack> & tracks)
  const
{
  if(algorithm=="gap"){// the present default
    if(zSep<1.0){
      return clusterize0(tracks);  
    }else{
      return clusterize1(tracks);  
    }

  }else if (algorithm=="DA"){

    vector< vector<reco::TransientTrack> > clusters;
    vector< TransientVertex > pv=vertices(tracks);
    if (pv.size()==0){ return clusters;}

    // fill into clusters and merge
    vector< reco::TransientTrack>  aCluster=pv.begin()->originalTracks();

    for(vector<TransientVertex>::iterator k=pv.begin()+1; k!=pv.end(); k++){
      
      if ( k->position().z() - (k-1)->position().z()> zSep ){
	// close a cluster
	clusters.push_back(aCluster);
	aCluster.clear();
      }
      for(unsigned int i=0; i<k->originalTracks().size(); i++){ aCluster.push_back( k->originalTracks().at(i)); }

    }
    clusters.push_back(aCluster);
    return clusters;
  }else{
    throw VertexException("TrackClusterizerInZ: unknown algorithm: " + algorithm);
  }
}

