// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzerMIPs
// 
//
// Original Author:  Pascal Paganini
//


// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>

//
// class declaration
//

class towerEner {   
 public:
  float eRec_, mean_, tpgGeV_;
  float data_[10] ;
  int tpgADC_,tpgEmul0_,tpgEmul1_,tpgEmul2_,tpgEmul3_ ,tpgEmul4_;
  int iphi_, ieta_, ttf_, fg_, nXtal_ ;
  float sample_ ;
  towerEner()
    : eRec_(0), mean_(0), tpgGeV_(0), tpgADC_(0),  
      tpgEmul0_(0),tpgEmul1_(0), tpgEmul2_(0), tpgEmul3_(0), tpgEmul4_(0), 
      iphi_(-999), ieta_(-999), ttf_(-999), fg_(-999), nXtal_(0), sample_(0)
  { 
    for (int i=0 ; i<10 ; i ++) data_[i] = 0. ; 
  }
};


class EcalTrigPrimAnalyzerMIPs : public edm::EDAnalyzer {
public:
  explicit EcalTrigPrimAnalyzerMIPs(const edm::ParameterSet&);
  ~EcalTrigPrimAnalyzerMIPs();
  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
private:

  TFile *histfile_;

  TTree *tree_ ;
  int iphi_, ieta_ , tpgADC_, ttf_, fg_, 
    tpgEmul0_, tpgEmul1_, tpgEmul2_, tpgEmul3_, tpgEmul4_,
    nevt_, nXtal_ ;
  float eRec_, mean_, tpgGeV_, sample_ ;
  float data0_, data1_, data2_, data3_, data4_, data5_, data6_, data7_, data8_, data9_ ;

  TTree *fedtree_ ;
  int fedId_, fedSize_ ;

  TTree * treetopbot_ ;
  int iphitop_, ietatop_, iphibot_, ietabot_, Ntop_, Nbot_ ;
  float Etop_, Ebot_ ;

  std::string label_;
  std::string producer_;
  std::string digi_label_;
  std::string digi_producer_;
  std::string emul_label_;
  std::string emul_producer_;

};

