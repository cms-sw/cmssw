#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"

using namespace std;


namespace {

  bool recTrackLessZ1(const track_t & tk1,
                     const track_t & tk2)
  {
    return tk1.z < tk2.z;
  }


  bool recTrackLessZ(const reco::TransientTrack & tk1,
                     const reco::TransientTrack & tk2)
  {
    return tk1.stateAtBeamLine().trackStateAtPCA().position().z() < tk2.stateAtBeamLine().trackStateAtPCA().position().z();
  }

}



// DA clusterizer
vector<track_t> TrackClusterizerInZ::fill(
			  const vector<reco::TransientTrack> & tracks
			  )const{
  // prepare track data
  vector<track_t> tks;
  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
    track_t t;
    t.z=((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan(((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    Measurement1D IP=(*it).stateAtBeamLine().transverseImpactParameter();
    t.ip=IP.value();
    t.dip=IP.error();

    //  get the beam-spot
    reco::BeamSpot beamspot=(it->stateAtBeamLine()).beamSpot();
    double x0,y0,wxy0;// can't make these members because of const
    x0=beamspot.x0();
    y0=beamspot.y0();
    wxy0=beamspot.BeamWidthX();

    t.dz2= pow((*it).track().dzError(),2)+pow(wxy0/tantheta,2);

    t.tt=&(*it);
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
  // evaluate assignment probabilities p(track,vertex), Z(track) and p(vertex)
  for(unsigned int i=0; i<nt; i++){

    tks[i].Z=0;
    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      double dx=tks[i].z-(*k).z;                      
      k->arg = beta*dx*dx/tks[i].dz2;                // beta d_ik^2
    }

    for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
      k->ptk[i] = k->py*exp(-k->arg); 
      tks[i].Z += k->ptk[i];                           // Z_i = sum_k exp(-beta d_ik^2)
    }
    // normalize the p_ik:  p_ik= exp(-beta d_ik^2) / sum_k' exp(-beta d_ik'^2) = exp(-beta)/Z_i
    if (tks[i].Z>0){
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	k->ptk[i]/=tks[i].Z;
      }
    }
  }

  // update vertex weights
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    double sump=0;
    for(unsigned int i=0; i<nt; i++){
      sump += k->ptk[i];
    }
    k->py=sump/nt;
  }

}





double TrackClusterizerInZ::fit(double beta,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{
  /* do a single fit at fixed temperature, assume that weights are up to date
     returns the squared sum of changes of vertex positions */
  if(DEBUG){cout << "TrackClusterizerInZ::fit" <<endl;}
  
  
  // "fit", get the weighted mean of track z for each vertex candidate
  double delta=0;
  unsigned int nt=tks.size();

  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    
    double sumwz=0;
    double sumw=0;
    for(unsigned int i=0; i<nt; i++){
      double w=k->ptk[i]/tks[i].dz2;  // weight * 1/sigma_z^2
      sumwz+=tks[i].z*w;
      sumw+=w;
    }
    double y1=sumwz/sumw; 
    delta+=pow(k->z-y1,2);
    k->z=y1;
    
    double sump=0;
    for(unsigned int i=0; i<nt; i++){
      sump+=k->ptk[i];
    }
    k->py=sump/nt;
    
  }// vertex loop

  return delta;
}


bool TrackClusterizerInZ::merge(vector<vertex_t> & y)const{
  // merge clusters that collapsed, shouldn't really be necessary
  if(y.size()<2)  return false;
  for(vector<vertex_t>::iterator k=y.begin(); (k+1)!=y.end(); k++){
    if (fabs((k+1)->z - k->z)<1.e-4){
      cout << "warning! merging cluster at " << k->z << endl;
      k->py += (k+1)->py;
      delete (k+1)->ptk;
      y.erase(k+1);
      return true;
    }
  }
  return false;
}







double TrackClusterizerInZ::split(
			 double beta,
			 vector<track_t> & tks,
			 vector<vertex_t> & y
			 )const{
  // split the vertex with the highest critical temperature, if it is above the current temperature
  // return a recommendation for the next temperature step (beta)
  
  unsigned int nt=tks.size();


  // calculate all critical temperatures, remember the one with the highest Tc
  vector<vertex_t>::iterator ksplit=y.end();  // means no splittable vertex
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    
    double a=0, b=0;
    double Ak=0, Bk=0;
    double Ck=0;
    for(unsigned int i=0; i<nt; i++){
      double dxi2=pow(tks[i].z-(k->z),2)/tks[i].dz2;
      double sig2=tks[i].dz2;
      double pik=k->ptk[i];
      a+=dxi2/sig2*pik;
      b+=pik/sig2;
      double u  =dxi2*beta;
      double aik=pik*(2.*u-1.)/sig2;
      double bik=pik*(4.*u*u-12.*u+3.)/6./sig2/sig2*beta;
      Ak+=-aik;
      Bk+=-bik+0.5*beta*aik*aik;
    }
    k->Tc=2.*a/b;  // the critical temperature of this vertex

    if (beta==0){      // no splitting at infinite Temperature
      k->epsilon=0;
      if ((ksplit==y.end()) || ( (ksplit!=y.end())  &&( k->Tc > ksplit->Tc))){
	ksplit=k;
      }

    }else{  // beta>0, finite Temperature

      if( k->Tc > (1/beta)){
	k->epsilon=sqrt(-0.5*Ak/Bk);  // split distance 
	if((Ak>0) || (Bk<0)){	 cout << "DEBUGME" << endl;  }
      }else{
	k->epsilon=0;
      }
      // should it be split?
      if ( (Ak<0) && (Bk>0) && (beta>0) && (k->Tc > (1/beta)) 
	   && (  (k==y.begin())  || ( (   k!=y.begin())   && ((k-1)->z <  k->z - k->epsilon)))
	   && (((k+1)==y.end())  || ( ((k+1)!=y.end() )   && ((k+1)->z >  k->z + k->epsilon)))
	   && ((ksplit==y.end()) || ( (ksplit!=y.end())  &&( k->Tc > ksplit->Tc)))
	   ){
	ksplit=k;
      }
    }
  }// vertex loop
  
  

  if (ksplit!=y.end()){
    if(beta==0){
      return 2./ksplit->Tc;
    }
    if(verbose_){std::cout << "splitting vertex at " << (*ksplit).z << "   by +/-" << (*ksplit).epsilon  <<std::endl;}
    vertex_t vnew;
    vnew.z       =(*ksplit).z-(*ksplit).epsilon;
    (*ksplit).z  =(*ksplit).z+(*ksplit).epsilon;
    vnew.py     = 0.5* (*ksplit).py;
    (*ksplit).py= 0.5* (*ksplit).py;
    vnew.ptk = new double[nt];
    for(unsigned int i=0; i<nt; i++){ vnew.ptk[i]=(*ksplit).ptk[i];}
    vnew.Tc=0;
    y.insert(ksplit,vnew);
    return beta;                 // don't update temperature after a split
  }else{
    return 2.*beta;               // in all other cases lower T by a factor 2
  }
}
 


TrackClusterizerInZ::TrackClusterizerInZ(const edm::ParameterSet& conf) 
{
  zSep = conf.getParameter<double>("zSeparation");
  betamax_=0.1; // default Tmin=10
  if(zSep<0){ betamax_=-1./zSep;} 
  maxIterations_=100;
  //verbose_=true;
  verbose_=false;
  //  verbose_= conf.getUntrackedParameter<bool>("verbose", false);
  DEBUG=false;
}


void TrackClusterizerInZ::dump(const double beta, vector<vertex_t> & y, const vector<track_t> & tks, int verbosity)const{
  cout << "-----TrackClusterizerInZ::dump ----" << endl;
  cout << "                                         z= ";
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    cout  <<  setw(9) << k->z ;
  }
  cout << "beta=" << beta << "   betamax= " << betamax_ << endl;
  cout << endl << "T=" << setw(15) << 1/beta <<"      Tc= ";
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    cout <<  setw(9) <<  k->Tc;
  }
  if(verbosity>0){
    cout << endl;
    cout << "----       z +/- dz            ip +/-dip    weights  ----" << endl;
    cout.precision(4);
    for(unsigned int i=0; i<tks.size(); i++){
      cout <<  setw (3)<< i << ")" <<  setw (8) << fixed << tks[i].z << " +/-" <<  setw (6)<< sqrt(tks[i].dz2);
      cout << "   " << setw (8) << tks[i].ip << " +/-" << setw (6) << tks[i].dip;
      for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
	cout <<  " " << setw (9) << k->ptk[i];
      }
      cout << endl;
    }
  }
  cout << endl << "----------" << endl;
}


vector< vector<reco::TransientTrack> > 
TrackClusterizerInZ::clusterizeDA(const vector<reco::TransientTrack> & tracks) 
const
{
  if(verbose_){cout << "running adaptive clustering " << endl;}


  vector<track_t> tks=fill(tracks);
  // sort in increasing order of z
  stable_sort(tks.begin(), tks.end(), recTrackLessZ1);
  if(verbose_){
    cout << "TrackClusterizerInZ::clusterize1  z-sorted tracklist" << endl;
    for(unsigned int i=0; i<tks.size(); i++){
      cout << i << " " << tks[i].z << " " << sqrt(tks[i].dz2) << endl;
    }
  }
  unsigned int nt=tracks.size();

  vector< vector<reco::TransientTrack> > clusters;
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
  
  double beta=0; 

  double delta=fit(beta, tks, y);
  if(verbose_){cout << "first fit: " << endl; dump(0,y,tks,1);}

  while(beta<betamax_){  // annealing loop, stop when T<Tmin  (i.e. beta>1/Tmin)

    beta=split(beta, tks, y);   // reduce temperature or split vertices
    updateWeights(beta, tks,y);

    if(verbose_) {cout << "after split split " << endl; dump(beta,y,tks,1);}

    // find equilibrium at new T
    delta=1.;
    int niter=0;      // number of iterations
    while((delta>1.e-5)  && (niter++ < maxIterations_)){ 

      delta=fit(beta,tks, y);
      updateWeights(beta, tks,y);

    }
    if (verbose_) {cout << "after fit fit   niter=" << niter << "  delta=" << delta << endl; dump(beta,y,tks,1);}
    while(merge(y)){if (verbose_) dump(beta,y,tks,1);}

  } // annealing loop, stop when T<Tmin


  // extract clusters, assign a track to a cluster if it has a weight>0.5
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){
    vector<reco::TransientTrack> cluster;
    cluster.clear();
    for(unsigned int i=0; i<nt; i++){
      if((k->ptk[i])>0.5){
	cluster.push_back(*(tks[i].tt));
      }
    }
    clusters.push_back(cluster);
  }
  // avoid a memory leak
  for(vector<vertex_t>::iterator k=y.begin(); k!=y.end(); k++){ delete k->ptk; }
  
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
      it++; if (it == tks.end()) break;
    }
  }

  // store last cluster
  clusters.push_back(currentCluster);

  return clusters;

}




vector< vector<reco::TransientTrack> >
TrackClusterizerInZ::clusterize(const vector<reco::TransientTrack> & tracks)
  const
{
  if(zSep>0){
    return clusterize0(tracks);  // the present default
  }else{
    return clusterizeDA(tracks); // DA clustering
  }
}

