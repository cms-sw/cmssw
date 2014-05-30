#ifndef L1TSECTOR_H
#define L1TSECTOR_H

#include <iostream>
#include <assert.h>
using namespace std;

#include "L1TRod.hh"
#include "L1TTracklets.hh"
#include "L1TTracks.hh"

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.



class L1TSector{

private:

  L1TSector():
    L11I_(0,1,1),
    L11O_(0,2,1),
    L12I_(0,1,2),
    L12O_(0,2,2),
    L13I_(0,1,3),
    L13O_(0,2,3),

    L31I_(0,3,1),
    L31O_(0,4,1),
    L32I_(0,3,2),
    L32O_(0,4,2),
    L33I_(0,3,3),
    L33O_(0,4,3),
    L34I_(0,3,4),
    L34O_(0,4,4),
    
    L5a1I_(0,5,1),
    L5a1O_(0,6,1),
    L5a2I_(0,5,2),
    L5a2O_(0,6,2),
    L5a3I_(0,5,3),
    L5a3O_(0,6,3),
    L5a4I_(0,5,4),
    L5a4O_(0,6,4),

    L5b1I_(0,7,1),
    L5b1O_(0,8,1),
    L5b2I_(0,7,2),
    L5b2O_(0,8,2),
    L5b3I_(0,7,3),
    L5b3O_(0,8,3),
    L5b4I_(0,7,4),
    L5b4O_(0,8,4),


    L51I_(0,9,1),
    L51O_(0,10,1),
    L52I_(0,9,2),
    L52O_(0,10,2),
    L53I_(0,9,3),
    L53O_(0,10,3),
    L54I_(0,9,4),
    L54O_(0,10,4),
    L55I_(0,9,5),
    L55O_(0,10,5),
    L56I_(0,9,6),
    L56O_(0,10,6),
    L57I_(0,9,7),
    L57O_(0,10,7)
  {
    n_=0;
  }

public:

  L1TSector(int n):
    L11I_(n,1,1),
    L11O_(n,2,1),
    L12I_(n,1,2),
    L12O_(n,2,2),
    L13I_(n,1,3),
    L13O_(n,2,3),

    L31I_(n,3,1),
    L31O_(n,4,1),
    L32I_(n,3,2),
    L32O_(n,4,2),
    L33I_(n,3,3),
    L33O_(n,4,3),
    L34I_(n,3,4),
    L34O_(n,4,4),
    
    L5a1I_(n,5,1),
    L5a1O_(n,6,1),
    L5a2I_(n,5,2),
    L5a2O_(n,6,2),
    L5a3I_(n,5,3),
    L5a3O_(n,6,3),
    L5a4I_(n,5,4),
    L5a4O_(n,6,4),

    L5b1I_(n,7,1),
    L5b1O_(n,8,1),
    L5b2I_(n,7,2),
    L5b2O_(n,8,2),
    L5b3I_(n,7,3),
    L5b3O_(n,8,3),
    L5b4I_(n,7,4),
    L5b4O_(n,8,4),


    L51I_(n,9,1),
    L51O_(n,10,1),
    L52I_(n,9,2),
    L52O_(n,10,2),
    L53I_(n,9,3),
    L53O_(n,10,3),
    L54I_(n,9,4),
    L54O_(n,10,4),
    L55I_(n,9,5),
    L55O_(n,10,5),
    L56I_(n,9,6),
    L56O_(n,10,6),
    L57I_(n,9,7),
    L57O_(n,10,7)
  {
    n_=n;
  }



  //uggly helper function
  void append(L1TTracklets& a,L1TTracklets b){
    for (unsigned int i=0;i<b.size();i++) {
      a.addTracklet(b.get(i));
    }
  }

  void findTracklets(int SL) {

    if (SL==1 || SL==0) {
      L1Tracklets_SL1=L11I_.findTracklets(L11O_);
    }

    if (SL==2 || SL==0) {
      // cout << "finding tracklets in SL=2"<<endl;
      L1Tracklets_SL2=L31I_.findTracklets(L31O_);
      append(L1Tracklets_SL2,L32I_.findTracklets(L32O_));
    }

    if (SL==3 || SL==0) {
      L1Tracklets_SL3=L51I_.findTracklets(L51O_);
      append(L1Tracklets_SL3,L52I_.findTracklets(L52O_));
      append(L1Tracklets_SL3,L53I_.findTracklets(L53O_));
      //int tmp=L1Tracklets_SL3.size();
      append(L1Tracklets_SL3,L5a1I_.findTracklets(L5a1O_));
      append(L1Tracklets_SL3,L5a2I_.findTracklets(L5a2O_));
      append(L1Tracklets_SL3,L5a3I_.findTracklets(L5a3O_));
      append(L1Tracklets_SL3,L5b1I_.findTracklets(L5b1O_));
      append(L1Tracklets_SL3,L5b2I_.findTracklets(L5b2O_));
      append(L1Tracklets_SL3,L5b3I_.findTracklets(L5b3O_));
      //cout << "In the short rings we found:"<<L1Tracklets_SL3.size()-tmp<<" tracklets"<<endl;
      //cout << "sizes:"
      //	   <<L5a1I_.nstubs()<<" "<<L5a1O_.nstubs()<<" "
      //   <<L5a2I_.nstubs()<<" "<<L5a2O_.nstubs()<<" "
      //   <<L5a3I_.nstubs()<<" "<<L5a3O_.nstubs()<<" "
      //   <<L5a4I_.nstubs()<<" "<<L5a4O_.nstubs()<<endl;
    }

  }


  unsigned int matchStubs(int SL) {
    
    unsigned int N_matches = 0;
    
    if (SL==1 || SL==0){
      N_matches += L31I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L32I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L33I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L34I_.matchTracklets(L1Tracklets_SL1);

      N_matches += L31O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L32O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L33O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L34O_.matchTracklets(L1Tracklets_SL1);

      N_matches += L5a1I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a2I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a3I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a4I_.matchTracklets(L1Tracklets_SL1);

      N_matches += L5b1I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b2I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b3I_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b4I_.matchTracklets(L1Tracklets_SL1);

      //N_matches += L51I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L52I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L53I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L54I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L55I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L56I_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L57I_.matchTracklets(L1Tracklets_SL1);

      N_matches += L5a1O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a2O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a3O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5a4O_.matchTracklets(L1Tracklets_SL1);

      N_matches += L5b1O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b2O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b3O_.matchTracklets(L1Tracklets_SL1);
      N_matches += L5b4O_.matchTracklets(L1Tracklets_SL1);

      //N_matches += L51O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L52O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L53O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L54O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L55O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L56O_.matchTracklets(L1Tracklets_SL1);
      //N_matches += L57O_.matchTracklets(L1Tracklets_SL1);
    }

    if (SL==2 || SL==0){
      N_matches += L11I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L12I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L13I_.matchTracklets(L1Tracklets_SL2);

      N_matches += L11O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L12O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L13O_.matchTracklets(L1Tracklets_SL2);

      N_matches += L5a1I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a2I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a3I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a4I_.matchTracklets(L1Tracklets_SL2);

      N_matches += L5b1I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b2I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b3I_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b4I_.matchTracklets(L1Tracklets_SL2);

      //N_matches += L51I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L52I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L53I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L54I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L55I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L56I_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L57I_.matchTracklets(L1Tracklets_SL2);

      N_matches += L5a1O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a2O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a3O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5a4O_.matchTracklets(L1Tracklets_SL2);

      N_matches += L5b1O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b2O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b3O_.matchTracklets(L1Tracklets_SL2);
      N_matches += L5b4O_.matchTracklets(L1Tracklets_SL2);


      //N_matches += L51O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L52O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L53O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L54O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L55O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L56O_.matchTracklets(L1Tracklets_SL2);
      //N_matches += L57O_.matchTracklets(L1Tracklets_SL2);
    }

    if (SL==3 || SL==0){
      N_matches += L11I_.matchTracklets(L1Tracklets_SL3);
      N_matches += L12I_.matchTracklets(L1Tracklets_SL3);
      N_matches += L13I_.matchTracklets(L1Tracklets_SL3);

      N_matches += L11O_.matchTracklets(L1Tracklets_SL3);
      N_matches += L12O_.matchTracklets(L1Tracklets_SL3);
      N_matches += L13O_.matchTracklets(L1Tracklets_SL3);

      N_matches += L31I_.matchTracklets(L1Tracklets_SL3);
      N_matches += L32I_.matchTracklets(L1Tracklets_SL3);
      N_matches += L33I_.matchTracklets(L1Tracklets_SL3);
      N_matches += L34I_.matchTracklets(L1Tracklets_SL3);

      N_matches += L31O_.matchTracklets(L1Tracklets_SL3);
      N_matches += L32O_.matchTracklets(L1Tracklets_SL3);
      N_matches += L33O_.matchTracklets(L1Tracklets_SL3);
      N_matches += L34O_.matchTracklets(L1Tracklets_SL3);
    }

    return N_matches;

  }

  unsigned int findCombinations(int SL) {
    unsigned int N_combs = 0;
    
    if (SL==1) {
      N_combs += L31I_.findCombinations(L1Tracklets_SL1);
      N_combs += L32I_.findCombinations(L1Tracklets_SL1);
      N_combs += L33I_.findCombinations(L1Tracklets_SL1);
      N_combs += L34I_.findCombinations(L1Tracklets_SL1);

      N_combs += L31O_.findCombinations(L1Tracklets_SL1);
      N_combs += L32O_.findCombinations(L1Tracklets_SL1);
      N_combs += L33O_.findCombinations(L1Tracklets_SL1);
      N_combs += L34O_.findCombinations(L1Tracklets_SL1);

      N_combs += L5a1I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a2I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a3I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a4I_.findCombinations(L1Tracklets_SL1);

      N_combs += L5b1I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b2I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b3I_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b4I_.findCombinations(L1Tracklets_SL1);

      N_combs += L51I_.findCombinations(L1Tracklets_SL1);
      N_combs += L52I_.findCombinations(L1Tracklets_SL1);
      N_combs += L53I_.findCombinations(L1Tracklets_SL1);
      N_combs += L54I_.findCombinations(L1Tracklets_SL1);
      N_combs += L55I_.findCombinations(L1Tracklets_SL1);
      N_combs += L56I_.findCombinations(L1Tracklets_SL1);
      N_combs += L57I_.findCombinations(L1Tracklets_SL1);

      N_combs += L5a1O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a2O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a3O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5a4O_.findCombinations(L1Tracklets_SL1);

      N_combs += L5b1O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b2O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b3O_.findCombinations(L1Tracklets_SL1);
      N_combs += L5b4O_.findCombinations(L1Tracklets_SL1);

      N_combs += L51O_.findCombinations(L1Tracklets_SL1);
      N_combs += L52O_.findCombinations(L1Tracklets_SL1);
      N_combs += L53O_.findCombinations(L1Tracklets_SL1);
      N_combs += L54O_.findCombinations(L1Tracklets_SL1);
      N_combs += L55O_.findCombinations(L1Tracklets_SL1);
      N_combs += L56O_.findCombinations(L1Tracklets_SL1);
      N_combs += L57O_.findCombinations(L1Tracklets_SL1);
    }

    if (SL==2) {
      N_combs += L11I_.findCombinations(L1Tracklets_SL2);
      N_combs += L12I_.findCombinations(L1Tracklets_SL2);
      N_combs += L13I_.findCombinations(L1Tracklets_SL2);

      N_combs += L11O_.findCombinations(L1Tracklets_SL2);
      N_combs += L12O_.findCombinations(L1Tracklets_SL2);
      N_combs += L13O_.findCombinations(L1Tracklets_SL2);

      N_combs += L5a1I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a2I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a3I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a4I_.findCombinations(L1Tracklets_SL2);

      N_combs += L5b1I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b2I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b3I_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b4I_.findCombinations(L1Tracklets_SL2);

      N_combs += L51I_.findCombinations(L1Tracklets_SL2);
      N_combs += L52I_.findCombinations(L1Tracklets_SL2);
      N_combs += L53I_.findCombinations(L1Tracklets_SL2);
      N_combs += L54I_.findCombinations(L1Tracklets_SL2);
      N_combs += L55I_.findCombinations(L1Tracklets_SL2);
      N_combs += L56I_.findCombinations(L1Tracklets_SL2);
      N_combs += L57I_.findCombinations(L1Tracklets_SL2);

      N_combs += L5a1O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a2O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a3O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5a4O_.findCombinations(L1Tracklets_SL2);

      N_combs += L5b1O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b2O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b3O_.findCombinations(L1Tracklets_SL2);
      N_combs += L5b4O_.findCombinations(L1Tracklets_SL2);

      N_combs += L51O_.findCombinations(L1Tracklets_SL2);
      N_combs += L52O_.findCombinations(L1Tracklets_SL2);
      N_combs += L53O_.findCombinations(L1Tracklets_SL2);
      N_combs += L54O_.findCombinations(L1Tracklets_SL2);
      N_combs += L55O_.findCombinations(L1Tracklets_SL2);
      N_combs += L56O_.findCombinations(L1Tracklets_SL2);
      N_combs += L57O_.findCombinations(L1Tracklets_SL2);
    }

    if (SL==3) {
      N_combs += L11I_.findCombinations(L1Tracklets_SL3);
      N_combs += L12I_.findCombinations(L1Tracklets_SL3);
      N_combs += L13I_.findCombinations(L1Tracklets_SL3);

      N_combs += L11O_.findCombinations(L1Tracklets_SL3);
      N_combs += L12O_.findCombinations(L1Tracklets_SL3);
      N_combs += L13O_.findCombinations(L1Tracklets_SL3);


      N_combs += L31I_.findCombinations(L1Tracklets_SL3);
      N_combs += L32I_.findCombinations(L1Tracklets_SL3);
      N_combs += L33I_.findCombinations(L1Tracklets_SL3);
      N_combs += L34I_.findCombinations(L1Tracklets_SL3);


      N_combs += L31O_.findCombinations(L1Tracklets_SL3);
      N_combs += L32O_.findCombinations(L1Tracklets_SL3);
      N_combs += L33O_.findCombinations(L1Tracklets_SL3);
      N_combs += L34O_.findCombinations(L1Tracklets_SL3);
    }


    return N_combs;
      
  }

  void findTracks() {

    for (unsigned int i=0; i<L1Tracklets_SL1.size(); i++) {
      L1TTracklet aTracklet = L1Tracklets_SL1.get(i);
      if (aTracklet.getStubs().size()>1) {
	L1TTrack aTrack(aTracklet);
	L1Tracks_.addTrack(aTrack);
      }
    }
    for (unsigned int i=0; i<L1Tracklets_SL2.size(); i++) {
      L1TTracklet aTracklet = L1Tracklets_SL2.get(i);
      if (aTracklet.getStubs().size()>1) {
	L1TTrack aTrack(aTracklet);
	L1Tracks_.addTrack(aTrack);
      }
    }
    for (unsigned int i=0; i<L1Tracklets_SL3.size(); i++) {
      L1TTracklet aTracklet = L1Tracklets_SL3.get(i);
      if (aTracklet.getStubs().size()>1) {
	L1TTrack aTrack(aTracklet);
	L1Tracks_.addTrack(aTrack);
      }
    }

  }

  L1TTracks getTracks() {
    return L1Tracks_;
  }

  void print() {

    cout << "Sector:"<<endl;

    cout << "SL1:" << L11I_.nstubs()<<" "<<L11O_.nstubs()
	 << " " << L12I_.nstubs()<<" "<<L12O_.nstubs()
	 << " " << L13I_.nstubs()<<" "<<L13O_.nstubs()
	 <<endl;
    cout << "SL2:" << L31I_.nstubs()<<" "<<L31O_.nstubs()
	 << " "<< L32I_.nstubs()<<" "<<L32O_.nstubs()
	 << " "<< L33I_.nstubs()<<" "<<L33O_.nstubs()
	 << " "<< L34I_.nstubs()<<" "<<L34O_.nstubs()
	 <<endl;
    cout << "SL3:" << L51I_.nstubs()<<" "<<L51O_.nstubs()
	 << " "<< L52I_.nstubs()<<" "<<L52O_.nstubs()
	 << " "<< L53I_.nstubs()<<" "<<L53O_.nstubs()
	 << " "<< L54I_.nstubs()<<" "<<L54O_.nstubs()
	 << " "<< L55I_.nstubs()<<" "<<L55O_.nstubs()
	 << " "<< L56I_.nstubs()<<" "<<L56O_.nstubs()
	 << " "<< L57I_.nstubs()<<" "<<L57O_.nstubs()
	 <<endl;

  }


  void printTracklets(int SL) {
    if (SL==1) {
      L1Tracklets_SL1.print();
      return;
    }
    if (SL==2) {
      L1Tracklets_SL2.print();
      return;
    }
    if (SL==3) {
      L1Tracklets_SL3.print();
      return;
    }
    if (SL==0) {
      L1TTracklets tmp = L1Tracklets_SL1;
      append(tmp,L1Tracklets_SL2);
      append(tmp,L1Tracklets_SL3);
      tmp.print();
      return;
    }
    assert(0);
    return;
  }

  L1TTracklets getTracklets(int SL) {
    if (SL==1){
      return L1Tracklets_SL1;
    }
    if (SL==2){
      return L1Tracklets_SL2;
    }
    if (SL==3){
      return L1Tracklets_SL3;
    }
    if (SL==0){
      L1TTracklets tmp = L1Tracklets_SL1;
      append(tmp,L1Tracklets_SL2);
      append(tmp,L1Tracklets_SL3);
      return tmp;
    }
    assert(0);
    return L1Tracklets_SL1; //will never get here, but compiler needs to 
                            //return something...

  }

  void addGeom(int layer,int contains,
	       int module,int ladder,double r1,double phi1,double r2,double phi2,
	       double phiSectorCenter){
    if(layer==1) {
      if (contains==1) L11I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L12I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L13I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==2) {
      if (contains==1) L11O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L12O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L13O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==3) {
      if (contains==1) L31I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L32I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L33I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L34I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==4) {
      if (contains==1) L31O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L32O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L33O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L34O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==5) {
      if (contains==1) L5a1I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L5a2I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L5a3I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L5a4I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==6) {
      if (contains==1) L5a1O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L5a2O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L5a3O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L5a4O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==7) {
      if (contains==1) L5b1I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L5b2I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L5b3I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L5b4I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==8) {
      if (contains==1) L5b1O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L5b2O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L5b3O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L5b4O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==9) {
      //cout << "n_="<<n_<<" layer==9 and contains="<<contains<<endl;
      if (contains==1) L51I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L52I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L53I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L54I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==5) L55I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==6) L56I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==7) L57I_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    else if (layer==10) {
      if (contains==1) L51O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==2) L52O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==3) L53O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==4) L54O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==5) L55O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==6) L56O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
      if (contains==7) L57O_.addGeom(module,ladder,r1,phi1,r2,phi2,phiSectorCenter);
    }
    
  }

  void addL11I(const L1TStub& aStub) {
    L11I_.addStub(aStub);
  }

  void addL11O(const L1TStub& aStub) {
    L11O_.addStub(aStub);
  }

  void addL12I(const L1TStub& aStub) {
    L12I_.addStub(aStub);
  }

  void addL12O(const L1TStub& aStub) {
    L12O_.addStub(aStub);
  }

  void addL13I(const L1TStub& aStub) {
    L13I_.addStub(aStub);
  }

  void addL13O(const L1TStub& aStub) {
    L13O_.addStub(aStub);
  }

  void addL31I(const L1TStub& aStub) {
    L31I_.addStub(aStub);
  }

  void addL31O(const L1TStub& aStub) {
    L31O_.addStub(aStub);
  }

  void addL32I(const L1TStub& aStub) {
    L32I_.addStub(aStub);
  }

  void addL32O(const L1TStub& aStub) {
    L32O_.addStub(aStub);
  }

  void addL33I(const L1TStub& aStub) {
    L33I_.addStub(aStub);
  }

  void addL33O(const L1TStub& aStub) {
    L33O_.addStub(aStub);
  }
  void addL34I(const L1TStub& aStub) {
    L34I_.addStub(aStub);
  }

  void addL34O(const L1TStub& aStub) {
    L34O_.addStub(aStub);
  }



  void addL5a1I(const L1TStub& aStub) {
    L5a1I_.addStub(aStub);
  }

  void addL5a1O(const L1TStub& aStub) {
    L5a1O_.addStub(aStub);
  }

  void addL5a2I(const L1TStub& aStub) {
    L5a2I_.addStub(aStub);
  }

  void addL5a2O(const L1TStub& aStub) {
    L5a2O_.addStub(aStub);
  }

  void addL5a3I(const L1TStub& aStub) {
    L5a3I_.addStub(aStub);
  }

  void addL5a3O(const L1TStub& aStub) {
    L5a3O_.addStub(aStub);
  }

  void addL5a4I(const L1TStub& aStub) {
    L5a4I_.addStub(aStub);
  }

  void addL5a4O(const L1TStub& aStub) {
    L5a4O_.addStub(aStub);
  }


  void addL5b1I(const L1TStub& aStub) {
    L5b1I_.addStub(aStub);
  }

  void addL5b1O(const L1TStub& aStub) {
    L5b1O_.addStub(aStub);
  }

  void addL5b2I(const L1TStub& aStub) {
    L5b2I_.addStub(aStub);
  }

  void addL5b2O(const L1TStub& aStub) {
    L5b2O_.addStub(aStub);
  }

  void addL5b3I(const L1TStub& aStub) {
    L5b3I_.addStub(aStub);
  }

  void addL5b3O(const L1TStub& aStub) {
    L5b3O_.addStub(aStub);
  }

  void addL5b4I(const L1TStub& aStub) {
    L5b4I_.addStub(aStub);
  }

  void addL5b4O(const L1TStub& aStub) {
    L5b4O_.addStub(aStub);
  }



  void addL51I(const L1TStub& aStub) {
    L51I_.addStub(aStub);
  }

  void addL51O(const L1TStub& aStub) {
    L51O_.addStub(aStub);
  }

  void addL52I(const L1TStub& aStub) {
    L52I_.addStub(aStub);
  }

  void addL52O(const L1TStub& aStub) {
    L52O_.addStub(aStub);
  }

  void addL53I(const L1TStub& aStub) {
    L53I_.addStub(aStub);
  }

  void addL53O(const L1TStub& aStub) {
    L53O_.addStub(aStub);
  }

  void addL54I(const L1TStub& aStub) {
    L54I_.addStub(aStub);
  }

  void addL54O(const L1TStub& aStub) {
    L54O_.addStub(aStub);
  }

  void addL55I(const L1TStub& aStub) {
    L55I_.addStub(aStub);
  }

  void addL55O(const L1TStub& aStub) {
    L55O_.addStub(aStub);
  }

  void addL56I(const L1TStub& aStub) {
    L56I_.addStub(aStub);
  }

  void addL56O(const L1TStub& aStub) {
    L56O_.addStub(aStub);
  }

  void addL57I(const L1TStub& aStub) {
    L57I_.addStub(aStub);
  }

  void addL57O(const L1TStub& aStub) {
    L57O_.addStub(aStub);
  }

  void printModuleMultiplicity() {
    L11I_.printModuleMultiplicity();
    L11O_.printModuleMultiplicity();

    L31I_.printModuleMultiplicity();
    L31O_.printModuleMultiplicity();
    L32I_.printModuleMultiplicity();
    L32O_.printModuleMultiplicity();

    L5a1I_.printModuleMultiplicity();
    L5a1O_.printModuleMultiplicity();
    L5a2I_.printModuleMultiplicity();
    L5a2O_.printModuleMultiplicity();

    L5b1I_.printModuleMultiplicity();
    L5b1O_.printModuleMultiplicity();
    L5b2I_.printModuleMultiplicity();
    L5b2O_.printModuleMultiplicity();

    L51I_.printModuleMultiplicity();
    L51O_.printModuleMultiplicity();
    L52I_.printModuleMultiplicity();
    L52O_.printModuleMultiplicity();
    L53I_.printModuleMultiplicity();
    L53O_.printModuleMultiplicity();
  }  

  void clean(){
    L1Tracklets_SL1.clean();
    L1Tracklets_SL2.clean();
    L1Tracklets_SL3.clean();

    L1Tracks_.clean();

    L11I_.clean();
    L11O_.clean();
    L12I_.clean();
    L12O_.clean();
    L13I_.clean();
    L13O_.clean();

    L31I_.clean();
    L31O_.clean();
    L32I_.clean();
    L32O_.clean();
    L33I_.clean();
    L33O_.clean();
    L34I_.clean();
    L34O_.clean();

    L5a1I_.clean();
    L5a1O_.clean();
    L5a2I_.clean();
    L5a2O_.clean();
    L5a3I_.clean();
    L5a3O_.clean();
    L5a4I_.clean();
    L5a4O_.clean();
   
    L5b1I_.clean();
    L5b1O_.clean();
    L5b2I_.clean();
    L5b2O_.clean();
    L5b3I_.clean();
    L5b3O_.clean();
    L5b4I_.clean();
    L5b4O_.clean();


    L51I_.clean();
    L51O_.clean();
    L52I_.clean();
    L52O_.clean();
    L53I_.clean();
    L53O_.clean();
    L54I_.clean();
    L54O_.clean();
    L55I_.clean();
    L55O_.clean();
    L56I_.clean();
    L56O_.clean();
    L57I_.clean();
    L57O_.clean();

  }

private:

  int n_; //sector number

  L1TTracklets L1Tracklets_SL1;
  L1TTracklets L1Tracklets_SL2;
  L1TTracklets L1Tracklets_SL3;

  L1TTracks L1Tracks_;

  L1TRod L11I_;
  L1TRod L11O_;
  L1TRod L12I_;
  L1TRod L12O_;
  L1TRod L13I_;
  L1TRod L13O_;

  L1TRod L31I_;
  L1TRod L31O_;
  L1TRod L32I_;
  L1TRod L32O_;
  L1TRod L33I_;
  L1TRod L33O_;
  L1TRod L34I_;
  L1TRod L34O_;

  L1TRod L5a1I_;
  L1TRod L5a1O_;
  L1TRod L5a2I_;
  L1TRod L5a2O_;
  L1TRod L5a3I_;
  L1TRod L5a3O_;
  L1TRod L5a4I_;
  L1TRod L5a4O_;

  L1TRod L5b1I_;
  L1TRod L5b1O_;
  L1TRod L5b2I_;
  L1TRod L5b2O_;
  L1TRod L5b3I_;
  L1TRod L5b3O_;
  L1TRod L5b4I_;
  L1TRod L5b4O_;


  L1TRod L51I_;
  L1TRod L51O_;
  L1TRod L52I_;
  L1TRod L52O_;
  L1TRod L53I_;
  L1TRod L53O_;
  L1TRod L54I_;
  L1TRod L54O_;
  L1TRod L55I_;
  L1TRod L55O_;
  L1TRod L56I_;
  L1TRod L56O_;
  L1TRod L57I_;
  L1TRod L57O_;

};



#endif



