#ifndef SLHCEVENT_H
#define SLHCEVENT_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <map>
#include <ext/hash_set>
#include <math.h>
#include <assert.h>
#include "L1TStub.hh"

#define NSECTORS 28

using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.


static double two_pi=8*atan(1.0);

static double x_offset=0.199196*0.0;
static double y_offset=0.299922*0.0;




class L1SimTrack{

public:

  L1SimTrack() {
   id_=-1; 
  }

  L1SimTrack(int id, int type, double pt, double eta, double phi, 
           double vx, double vy, double vz) {
    id_=id;
    type_=type;
    pt_=pt;
    eta_=eta;
    phi_=phi;
    vx_=vx;
    vy_=vy;
    vz_=vz;
  }

  void write(ofstream& out){
    
    out << "SimTrack: " 
	<< id_ << "\t" 
	<< type_ << "\t" 
	<< pt_ << "\t" 
	<< eta_ << "\t" 
	<< phi_ << "\t" 
	<< vx_ << "\t" 
	<< vy_ << "\t" 
	<< vz_ << "\t" << endl; 
	
  }
  void write(ostream& out){
    
    out << "SimTrack: " 
	<< id_ << "\t" 
	<< type_ << "\t" 
	<< pt_ << "\t" 
	<< eta_ << "\t" 
	<< phi_ << "\t" 
	<< vx_ << "\t" 
	<< vy_ << "\t" 
	<< vz_ << "\t" << endl; 
	
  }
  
  int id() const { return id_; }
  int type() const { return type_; }
  double pt() { return pt_; }
  double eta() { return eta_; }
  double phi() { return phi_; }
  double vx() { return vx_; }
  double vy() { return vy_; }
  double vz() { return vz_; }

private:

  int id_;
  int type_;
  double pt_;
  double eta_;
  double phi_;
  double vx_;
  double vy_;
  double vz_;

};


class Digi{

public:


  Digi(int layer,int irphi, int iz, int sensorlayer,
       int ladder, int module, double x, double y, double z) {
    layer_=layer;
    irphi_=irphi;
    iz_=iz;
    sensorlayer_=sensorlayer;
    ladder_=ladder;
    module_=module;
    x_=x;
    y_=y;
    z_=z;
  }

  void AddSimtrack(int simtrackid){
    simtrackids_.push_back(simtrackid);
  }

  void write(ofstream& out){
    
    out << "Digi: " 
	<< layer_ << "\t" 
	<< irphi_ << "\t" 
	<< iz_ << "\t" 
	<< sensorlayer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" << endl; 

    for (unsigned int i=0;i<simtrackids_.size();i++){
      out << "SimTrackId: "<<simtrackids_[i]<<endl;
    }
	
  }
  void write(ostream& out){
    
    out << "Digi: " 
	<< layer_ << "\t" 
	<< irphi_ << "\t" 
	<< iz_ << "\t" 
	<< sensorlayer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" << endl; 

    for (unsigned int i=0;i<simtrackids_.size();i++){
      out << "SimTrackId: "<<simtrackids_[i]<<endl;
    }
	
  }

  int irphi() {return irphi_;}
  int iz() {return iz_;}
  int layer() {return layer_;}
  int sensorlayer() {return sensorlayer_;}
  int ladder() {return ladder_;}
  int module() {return module_;}
  double r() {return sqrt(x_*x_+y_*y_);}
  double z() {return z_;}
  double phi() {return atan2(y_,x_);}


  bool operator==(const Digi& anotherdigi) const {
    if (irphi_!=anotherdigi.irphi_) return false;
    if (iz_!=anotherdigi.iz_) return false;
    if (layer_!=anotherdigi.layer_) return false;
    if (ladder_!=anotherdigi.ladder_) return false;
    return module_==anotherdigi.module_;    
  }

  int hash() const {
    return irphi_+iz_*1009+layer_*10000003+ladder_*1000003+module_*10007;
  }

  int nsimtrack() {return simtrackids_.size();}
  int simtrackid(int isim) {return simtrackids_[isim];}
  bool matchsimtrackid(int simtrackid){
    for (unsigned int i=0;i<simtrackids_.size();i++){
      if (simtrackids_[i]==simtrackid) return true;
    }
    return false;
  }


private:

  unsigned int layer_;
  unsigned int ladder_;
  unsigned int module_;
  int irphi_;
  int iz_;
  int sensorlayer_;
  double x_;
  double y_;
  double z_;

  vector<int> simtrackids_;

};

struct HashOp {
  int operator()(const Digi &a) const {
    return a.hash();
  }
};
 
struct HashEqual {
  bool operator()(const Digi &a, const Digi &b) const {
    return a == b;
  }
};




class SLHCEvent{

public:


  SLHCEvent() {
    //empty constructor to be used with 'filler' functions
    eventnum_=0;
  }

  void setIPx(double x) { x_offset=x;}
  void setIPy(double y) { y_offset=y;}

  void addL1SimTrack(int id,int type,double pt,double eta,double phi,
	      double vx,double vy,double vz){

    vx-=x_offset;
    vy-=y_offset;
    L1SimTrack simtrack(id,type,pt,eta,phi,vx,vy,vz);
    simtracks_.push_back(simtrack);

  }


  void addDigi(int layer,int irphi,int iz,int sensorlayer,int ladder,int module,
	  double x,double y,double z,vector<int> simtrackids){

    x-=x_offset;
    y-=y_offset;

    Digi digi(layer,irphi,iz,sensorlayer,ladder,
	      module,x,y,z);

    for (unsigned int i=0;i<simtrackids.size();i++){
      digi.AddSimtrack(simtrackids[i]);
    }    
  
    digis_.push_back(digi);
    digihash_.insert(digi);

  }


  bool addStub(int layer,int ladder,int module, int strip, double pt,double bend,
	   double x,double y,double z,
	   vector<bool> innerStack,
	   vector<int> irphi,
	   vector<int> iz,
	   vector<int> iladder,
	   vector<int> imodule){

    x-=x_offset;
    y-=y_offset;

    L1TStub stub(-1,-1,-1,layer, ladder, module, strip, 
		 x, y, z, -1.0, -1.0, pt, bend);

    for(unsigned int i=0;i<innerStack.size();i++){
      if (innerStack[i]) {
	stub.AddInnerDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
      else {
	stub.AddOuterDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
    }   

    bool foundclose=false;

    for (unsigned int i=0;i<stubs_.size();i++) {
      if (fabs(stubs_[i].x()-stub.x())<0.02&&
	  fabs(stubs_[i].y()-stub.y())<0.02&&
	  fabs(stubs_[i].z()-stub.z())<0.2) {
	foundclose=true;
      }
    }

    stub.setiphi(stub.diphi());
    stub.setiz(stub.diz());

    
    if (!foundclose) {
      stubs_.push_back(stub);
      return true;
    }

    return false;
    
  }

  L1TStub lastStub(){
    return stubs_.back();
  }

  SLHCEvent(istream& in) {

    string tmp;
    in >> tmp;
    while (tmp=="Map:") {
      in>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp;
      in>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp;
    }
    if (tmp=="EndMap") {
      in>>tmp;
    }
    if (tmp!="Event:") {
      cout << "Expected to read 'Event:' but found:"<<tmp<<endl;
      abort();
    }
    in >> eventnum_;

    //cout << "Started to read event="<<eventnum_<<endl;

    // read the SimTracks

    bool first=true;

    in >> tmp;
    while (tmp!="SimTrackEnd"){
      if (!(tmp=="SimTrack:"||tmp=="SimTrackEnd")) {
	cout << "Expected to read 'SimTrack:' or 'SimTrackEnd' but found:"
	     << tmp << endl;
	abort();
      }
      int id;
      int type;
      double pt;
      double eta;
      double phi;
      double vx;
      double vy;
      double vz;
      in >> id >> type >> pt >> eta >> phi >> vx >> vy >> vz;
      if (first) {
	//mc_rinv=0.00299792*3.8/pt;
	//mc_phi0=phi;
	//mc_z0=vz;
	//double two_pi=8*atan(1.0);
	//mc_t=tan(0.25*two_pi-2.0*atan(exp(-eta)));
	//event=eventnum_;
	first=false;
      }
      vx-=x_offset;
      vy-=y_offset;
      L1SimTrack simtrack(id,type,pt,eta,phi,vx,vy,vz);
      simtracks_.push_back(simtrack);
      in >> tmp;
    }


   
    //read te Digis
    in >> tmp;
    while (tmp!="DigiEnd"){
      if (!(tmp=="Digi:"||tmp=="DigiEnd")) {
	cout << "Expected to read 'Digi:' or 'DigiEnd' but found:"
	     << tmp << endl;
        abort();
      }
      int layer;
      int irphi;
      int iz;
      int sensorlayer;
      int ladder;
      int module;
      double x;
      double y;
      double z;

      in >> layer
	 >> irphi
	 >> iz
	 >> sensorlayer
	 >> ladder
	 >> module
	 >> x
	 >> y
	 >> z;

      x-=x_offset;
      y-=y_offset;


      Digi digi(layer,irphi,iz,sensorlayer,ladder,
		module,x,y,z);
      in >> tmp;
      while (tmp=="SimTrackId:"){
	int simtrackid;
	in >> simtrackid;
	digi.AddSimtrack(simtrackid);
	in >> tmp;
      }      
      digis_.push_back(digi);
      digihash_.insert(digi);
    }

    //cout << "Read "<<digis_.size()<<" digis"<<endl;

    int nlayer[11];
    for (int i=0;i<10;i++) {
      nlayer[i]=0;
    }
    

    //read stubs
    in >> tmp;
    while (tmp!="StubEnd"){

      if (!in.good()) {
	cout << "File not good"<<endl;
	abort();
      };
      if (!(tmp=="Stub:"||tmp=="StubEnd")) {
	cout << "Expected to read 'Stub:' or 'StubEnd' but found:"
	     << tmp << endl;
	abort();
      }
      int layer;
      int ladder;
      int module;
      int simtrk;
      int strip;
      double pt;
      double x;
      double y;
      double z;
      double bend;

      in >> layer >> ladder >> module >> strip >> simtrk >> pt >> x >> y >> z >> bend;

      layer--;   
      x-=x_offset;
      y-=y_offset;

      if (layer < 10) nlayer[layer]++;

      L1TStub stub(-1,-1,-1,layer, ladder, module, strip, x, y, z, -1.0, -1.0, pt, bend);

      in >> tmp;

      while (tmp=="InnerStackDigi:"||tmp=="OuterStackDigi:"){
	int irphi;
	int iz;
        int iladder;
        int imodule;
	in >> irphi;
	in >> iz;
	in >> iladder; 
        in >> imodule;
	if (tmp=="InnerStackDigi:") stub.AddInnerDigi(iladder,imodule,irphi,iz);
	if (tmp=="OuterStackDigi:") stub.AddOuterDigi(iladder,imodule,irphi,iz);
	in >> tmp;
      }   

      bool foundclose=false;

      for (unsigned int i=0;i<stubs_.size();i++) {
	if (fabs(stubs_[i].x()-stub.x())<0.02&&
	    fabs(stubs_[i].y()-stub.y())<0.02&&
	    fabs(stubs_[i].z()-stub.z())<0.2) {
	  foundclose=true;
	}
      }

      /*
      double t=fabs(stub.z())/stub.r();
      static double piovertwo=2.0*atan(1);
      double theta=piovertwo-atan(t);
      double eta=-log(tan(0.5*theta));
      */

      //if (!foundclose&&(fabs(eta)<2.6)) {
      if (!foundclose) {
	stubs_.push_back(stub);
      }
    }
    //cout << "Read "<<stubs_.size()<<" stubs"<<endl;

  }

  void write(ofstream& out){
    
    out << "Event: "<<eventnum_ << endl;
      
    for (unsigned int i=0; i<simtracks_.size(); i++) {
      simtracks_[i].write(out);
    }
    out << "SimTrackEnd" << endl;
    
    for (unsigned int i=0; i<digis_.size(); i++) {
      digis_[i].write(out);
    }
    out << "DigiEnd" << endl;

    for (unsigned int i=0; i<stubs_.size(); i++) {
      stubs_[i].write(out);
    }
    out << "StubEnd" << endl;
    
  }

  void write(ostream& out){
    
    out << "Event: "<<eventnum_ << endl;
      
    for (unsigned int i=0; i<simtracks_.size(); i++) {
      simtracks_[i].write(out);
    }
    out << "SimTrackEnd" << endl;
    
    for (unsigned int i=0; i<digis_.size(); i++) {
      digis_[i].write(out);
    }
    out << "DigiEnd" << endl;

    for (unsigned int i=0; i<stubs_.size(); i++) {
      stubs_[i].write(out);
    }
    out << "StubEnd" << endl;
    
  }


  int simtrackid(const L1TStub& stub){

    std::vector<int> simtrackids;

    simtrackids=this->simtrackids(stub);

    if (simtrackids.size()==0) {
      return -1;
    }


    std::sort(simtrackids.begin(),simtrackids.end());

    int n_max = 0;
    int value_max = 0;
    int n_tmp = 1;
    int value_tmp = simtrackids[0];
    for (unsigned int i=1; i<simtrackids.size();i++) {
      if (simtrackids[i] == value_tmp) n_tmp++;
      else {
	if (n_tmp > n_max) {
	  n_max = n_tmp;
	  value_max = value_tmp;
	}
	n_tmp = 1;
	value_tmp = simtrackids[i];
      }
    }
    
    if (n_tmp > n_max) value_max = value_tmp;

    return value_max;

  }

  std::vector<int> simtrackids(const L1TStub& stub){

    //cout << "Entering simtrackids"<<endl;

    std::vector<int> simtrackids;

    int layer=stub.layer()+1;


    vector<pair<int,int> > innerdigis=stub.innerdigis();
    vector<pair<int,int> > outerdigis=stub.outerdigis();
    vector<pair<int,int> > innerdigisladdermodule=stub.innerdigisladdermodule();
    vector<pair<int,int> > outerdigisladdermodule=stub.outerdigisladdermodule();

    vector<pair<int,int> > alldigis=stub.innerdigis();
    alldigis.insert(alldigis.end(),outerdigis.begin(),outerdigis.end());
    vector<pair<int,int> > alldigisladdermodule=stub.innerdigisladdermodule();
    alldigisladdermodule.insert(alldigisladdermodule.end(),
				outerdigisladdermodule.begin(),
				outerdigisladdermodule.end());



    if (layer<1000) {

      for (unsigned int k=0;k<alldigis.size();k++){
	int irphi=alldigis[k].first;
	int iz=alldigis[k].second;
	int ladder=alldigisladdermodule[k].first;
	int module=alldigisladdermodule[k].second;
	Digi tmp(layer,irphi,iz,-1,ladder,module,0.0,0.0,0.0);
	__gnu_cxx::hash_set<Digi,HashOp,HashEqual>::const_iterator it=digihash_.find(tmp);
	if(it==digihash_.end()){
	  static int count=0;
	  count++;
	  if (count<0) {
	    cout << "Warning did not find digi"<<endl;
	  } 
 	}
	else{
	  Digi adigi=*it;
	  for(int idigi=0;idigi<adigi.nsimtrack();idigi++){
	    simtrackids.push_back(adigi.simtrackid(idigi));
	  }
	}	
      }
    }

    else{

      for (unsigned int k=0;k<alldigis.size();k++){
	int irphi=alldigis[k].first;
	int iz=alldigis[k].second;
	int module=alldigisladdermodule[k].second;
	int offset=1000;
	if (stub.z()<0.0) offset=2000;
	Digi tmp(stub.module()+offset,irphi,iz,-1,1,module,0.0,0.0,0.0);
	__gnu_cxx::hash_set<Digi,HashOp,HashEqual>::const_iterator it=digihash_.find(tmp);
	if(it==digihash_.end()){
	  static int count=0;
	  count++;
	  if (count < 0) {
	    cout << "Warning did not find digi in disks"<<endl;
	  }
	}
	else{
	  //cout << "Warning found digi in disks"<<endl;
	  Digi adigi=*it;
	  for(int idigi=0;idigi<adigi.nsimtrack();idigi++){
	    simtrackids.push_back(adigi.simtrackid(idigi));
	  }
	}	
      }
    }

    return simtrackids;

  }

  int ndigis() { return digis_.size(); }

  Digi digi(int i) { return digis_[i]; }

  int nstubs() { return stubs_.size(); }

  L1TStub stub(int i) { return stubs_[i]; }

  int nsimtracks() { return simtracks_.size(); }

  L1SimTrack simtrack(int i) { return simtracks_[i]; }

  int eventnum() const { return eventnum_; }

  int getSimtrackFromSimtrackid(int simtrackid) const {
    for(unsigned int i=0;i<simtracks_.size();i++){
      if (simtracks_[i].id()==simtrackid) return i;
    }
    return -1;
  }



  //static double mc_rinv;
  //static double mc_phi0;
  //static double mc_z0;
  //static double mc_t;
  //static int event;

private:

  int eventnum_;
  vector<L1SimTrack> simtracks_;
  vector<Digi> digis_;
  __gnu_cxx::hash_set<Digi,HashOp,HashEqual> digihash_;
  vector<L1TStub> stubs_;


};

//double SLHCEvent::mc_rinv=0.0;
//double SLHCEvent::mc_phi0=0.0;
//double SLHCEvent::mc_z0=0.0;
//double SLHCEvent::mc_t=0.0;
//int SLHCEvent::event=0;

#endif



