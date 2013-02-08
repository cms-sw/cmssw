#ifndef L1TTRACKLET_H
#define L1TTRACKLET_H

#include <iostream>
#include <map>
#include <assert.h>

using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.



class L1TTracklet{

public:

  L1TTracklet(int innerStub, int outerStub, 
	      double rinv, double phi0, double z0, double t, 
	      int irinv, int iphi0, int iz0, int it,
	      int layer, int local_rod, double sectorCenter){
    
    innerStub_=innerStub;
    outerStub_=outerStub;

    rinv_=rinv;
    phi0_=phi0;
    z0_=z0;
    t_=t;

    irinv_=irinv;
    iphi0_=iphi0;
    iz0_=iz0;
    it_=it;

    sectorCenter_=sectorCenter;

    layer_=layer;
    local_rod_=local_rod;

  }

  L1TTracklet(){
  }

  void addStub(const L1TStub& j){
    stubs_.push_back(j);
  }

  vector<L1TStub> getStubs() const {
    return stubs_;
  }

  vector<L1TStub> getStubComponents() {
    return stubComponents_;
  }

  void addStubComponent(const L1TStub& j){
    stubComponents_.push_back(j);
  }

  vector<L1TStub> getAllStubs() const {
    vector<L1TStub> tmp;
    for (unsigned int i=0; i<stubComponents_.size(); i++){
      tmp.push_back(stubComponents_[i]);
    }
    for (unsigned int j=0; j<stubs_.size(); j++){
      tmp.push_back(stubs_[j]);
    }
    return tmp;
  }

  int getSuperLayer() {
    if (stubComponents_[0].layer()==1 || stubComponents_[0].layer()==2)
      return 1;
    else if  (stubComponents_[0].layer()==3 || stubComponents_[0].layer()==4)
      return 2;
    else if  (stubComponents_[0].layer()==9 || stubComponents_[0].layer()==10)
      return 3;
    else return -999;

    assert(0);
    return -999;
  }

  void print() {

    if (1) {

      static ofstream out("stubfinding.txt");

      out <<rinv_<<" "<<phi0_<<" "<<t_<<" "<<innerStub_<<" "<<outerStub_<<" "<<stubs_.size()<<endl;

    }

    cout << "Tracklet: "<<rinv_<<" "<<phi0_<<" "<<t_<<" "<<innerStub_<<" "<<outerStub_<<" "<<stubs_.size()<<endl;

  }

  double rinv() const {return rinv_; }
  double phi0() const {return phi0_; }
  double z0() const {return z0_; }
  double t() const {return t_; }

  int irinv() const {return irinv_; }
  int iphi0() const {return iphi0_; }
  int iz0() const {return iz0_; }
  int it() const {return it_; }

  int nMatchedStubs() const {return stubs_.size(); }

  void makeMatchedStubsMap() {
    int nSL1 = 0;
    int nSL2 = 0;
    int nSL3 = 0;
    for (unsigned int j=0; j<stubs_.size(); j++) {
      if (stubs_[j].layer()==1 || stubs_[j].layer()==2)
        nSL1++;
      else if (stubs_[j].layer()==3 || stubs_[j].layer()==4)
        nSL2++;
      else if (stubs_[j].layer()==9 || stubs_[j].layer()==10)
        nSL3++;
    }
    nMatchedStubs_.insert(make_pair(1,nSL1));
    nMatchedStubs_.insert(make_pair(2,nSL2));
    nMatchedStubs_.insert(make_pair(3,nSL3));
  }

  int layer() {
    return layer_;
  }

  int local_rod() {
    return local_rod_;
  }

  inline unsigned int trkletMultiplSL1() {return nMatchedStubs_[1];}
  inline unsigned int trkletMultiplSL2() {return nMatchedStubs_[2];}
  inline unsigned int trkletMultiplSL3() {return nMatchedStubs_[3];}

  double r() { return stubComponents_[0].r(); }
  double z() { return stubComponents_[0].z(); }

  double sectorCenter() const { return sectorCenter_; }

private:

  int layer_;
  int local_rod_;

  int innerStub_;
  int outerStub_;
  double rinv_;
  double phi0_;
  double z0_;
  double t_;

  double irinv_;
  double iphi0_;
  double iz0_;
  double it_;

  double sectorCenter_;

  vector<L1TStub> stubs_;
  vector<L1TStub> stubComponents_;
  map<int, int> nMatchedStubs_;


};



#endif



