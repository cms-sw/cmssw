#ifndef MultiTrackValidatorBase_h
#define MultiTrackValidatorBase_h

/** \class MultiTrackValidatorBase
 *  Base class for analyzers that produces histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2008/03/25 14:50:33 $
 *  $Revision: 1.6 $
 *  \author cerati
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/TrackingParticleSelector.h"

#include <iostream>
#include <sstream>
#include <string>
#include <TH1F.h>
#include <TH2F.h>

class MultiTrackValidatorBase {
 public:
  /// Constructor
  MultiTrackValidatorBase(const edm::ParameterSet& pset):
    sim(pset.getParameter<std::string>("sim")),
    label(pset.getParameter< std::vector<edm::InputTag> >("label")),
    bsSrc(pset.getParameter< edm::InputTag >("beamSpot")),
    label_tp_effic(pset.getParameter< edm::InputTag >("label_tp_effic")),
    label_tp_fake(pset.getParameter< edm::InputTag >("label_tp_fake")),
    associators(pset.getParameter< std::vector<std::string> >("associators")),
    associatormap(pset.getParameter< edm::InputTag >("associatormap")),
    UseAssociators(pset.getParameter< bool >("UseAssociators")),
    out(pset.getParameter<std::string>("out")),

    tpSelector(pset.getParameter<double>("ptMinTP"),
	       pset.getParameter<double>("minRapidityTP"),
	       pset.getParameter<double>("maxRapidityTP"),
	       pset.getParameter<double>("tipTP"),
	       pset.getParameter<double>("lipTP"),
	       pset.getParameter<int>("minHitTP"),
	       pset.getParameter<bool>("signalOnlyTP"),
	       //pset.getParameter<bool>("chargedOnlyTP"),
	       pset.getParameter<std::vector<int> >("pdgIdTP")),
    
    min(pset.getParameter<double>("min")),
    max(pset.getParameter<double>("max")),
    nint(pset.getParameter<int>("nint")),
    useFabs(pset.getParameter<bool>("useFabsEta")),
    minpT(pset.getParameter<double>("minpT")),
    maxpT(pset.getParameter<double>("maxpT")),
    nintpT(pset.getParameter<int>("nintpT")),
    minHit(pset.getParameter<double>("minHit")),
    maxHit(pset.getParameter<double>("maxHit")),
    nintHit(pset.getParameter<int>("nintHit")),
    useInvPt(pset.getParameter<bool>("useInvPt"))
    {
      dbe_ = edm::Service<DQMStore>().operator->();
    }
  
  /// Destructor
  virtual ~MultiTrackValidatorBase(){ }
  
  virtual void doProfileX(TH2 * th2, MonitorElement* me){
    if (th2->GetNbinsX()==me->getNbinsX()){
      TH1F * h1 = (TH1F*) th2->ProfileX();
      for (int bin=0;bin!=h1->GetNbinsX();bin++){
	me->setBinContent(bin+1,h1->GetBinContent(bin+1));
	me->setBinError(bin+1,h1->GetBinError(bin+1));
      }
      delete h1;
    } else {
      throw cms::Exception("MultiTrackValidator") << "Different number of bins!";
    }
  }

  virtual void doProfileX(MonitorElement * th2m, MonitorElement* me) {
    doProfileX(th2m->getTH2F(), me);
  }

  virtual double getEta(double eta) {
    if (useFabs) return fabs(eta);
    else return eta;
  }

  virtual double getPt(double pt) {
    if (useInvPt && pt!=0) return 1/pt;
    else return pt;
  }
  
  void fillPlotFromVector(MonitorElement* h, std::vector<int>& vec) {
    for (unsigned int j=0; j<vec.size(); j++){
      h->setBinContent(j+1, vec[j]);
    }
  }

  void fillPlotFromVectors(MonitorElement* h, std::vector<int>& numerator, std::vector<int>& denominator,std::string type){
    double value,err;
    for (unsigned int j=0; j<numerator.size(); j++){
      if (denominator[j]!=0){
	if (type=="effic")
	  value = ((double) numerator[j])/((double) denominator[j]);
	else if (type=="fakerate")
	  value = 1-((double) numerator[j])/((double) denominator[j]);
	else return;
	err = sqrt( value*(1-value)/(double) denominator[j] );
	h->setBinContent(j+1, value);
	h->setBinError(j+1,err);
      }
      else {
	h->setBinContent(j+1, 0);
      }
    }
  }

  void setUpVectors() {
    std::vector<double> etaintervalsv;
    std::vector<double> pTintervalsv;
    std::vector<int>    totSIMveta,totASSveta,totASS2veta,totRECveta;
    std::vector<int>    totSIMvpT,totASSvpT,totASS2vpT,totRECvpT;
    std::vector<int>    totSIMv_hit,totASSv_hit,totASS2v_hit,totRECv_hit;

    double step=(max-min)/nint;
    std::ostringstream title,name;
    etaintervalsv.push_back(min);
    for (int k=1;k<nint+1;k++) {
      double d=min+k*step;
      etaintervalsv.push_back(d);
      totSIMveta.push_back(0);
      totASSveta.push_back(0);
      totASS2veta.push_back(0);
      totRECveta.push_back(0);
    }   
    etaintervals.push_back(etaintervalsv);
    totSIMeta.push_back(totSIMveta);
    totASSeta.push_back(totASSveta);
    totASS2eta.push_back(totASS2veta);
    totRECeta.push_back(totRECveta);
  
    double steppT = (maxpT-minpT)/nintpT;
    pTintervalsv.push_back(minpT);
    for (int k=1;k<nintpT+1;k++) {
      double d=minpT+k*steppT;
      pTintervalsv.push_back(d);
      totSIMvpT.push_back(0);
      totASSvpT.push_back(0);
      totASS2vpT.push_back(0);
      totRECvpT.push_back(0);
    }
    pTintervals.push_back(pTintervalsv);
    totSIMpT.push_back(totSIMvpT);
    totASSpT.push_back(totASSvpT);
    totASS2pT.push_back(totASS2vpT);
    totRECpT.push_back(totRECvpT);

    for (int k=1;k<nintHit+1;k++) {
      totSIMv_hit.push_back(0);
      totASSv_hit.push_back(0);
      totASS2v_hit.push_back(0);
      totRECv_hit.push_back(0);
    }
    totSIM_hit.push_back(totSIMv_hit);
    totASS_hit.push_back(totASSv_hit);
    totASS2_hit.push_back(totASS2v_hit);
    totREC_hit.push_back(totRECv_hit);
  }

 protected:

  DQMStore* dbe_;

  std::string sim;
  std::vector<edm::InputTag> label;
  edm::InputTag bsSrc;
  edm::InputTag label_tp_effic;
  edm::InputTag label_tp_fake;
  std::vector<std::string> associators;
  edm::InputTag associatormap;
  bool UseAssociators;
  std::string out;
  
  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      
	       
  double  min, max;
  int nint;
  bool useFabs;
  double minpT, maxpT;
  int nintpT;
  double minHit, maxHit;
  int nintHit;
  bool useInvPt;

  edm::ESHandle<MagneticField> theMF;
  std::vector<const TrackAssociatorBase*> associator;

  //sim
  std::vector<MonitorElement*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;

  //1D
  std::vector<MonitorElement*> h_tracks, h_fakes, h_hits, h_charge;
  std::vector<MonitorElement*> h_effic, h_efficPt, h_fakerate, h_fakeratePt, h_recoeta, h_assoceta, h_assoc2eta, h_simuleta;
  std::vector<MonitorElement*>  h_effic_vs_hit, h_fake_vs_hit;
  std::vector<MonitorElement*> h_recopT, h_assocpT, h_assoc2pT, h_simulpT;
  std::vector<MonitorElement*> h_pt, h_eta, h_pullTheta,h_pullPhi,h_pullDxy,h_pullDz,h_pullQoverp;

  //2D  
  std::vector<MonitorElement*> nrec_vs_nsim;

  //assoc hits
  std::vector<MonitorElement*> h_assocFraction, h_assocSharedHit;

  //#hit vs eta: to be used with doProfileX
  std::vector<MonitorElement*> nhits_vs_eta;
  std::vector<MonitorElement*> h_hits_eta;
 
  std::vector< std::vector<double> > etaintervals;
  std::vector< std::vector<double> > pTintervals;
  std::vector< std::vector<int> > totSIMeta,totRECeta,totASSeta,totASS2eta;
  std::vector< std::vector<int> > totSIMpT,totRECpT,totASSpT,totASS2pT;
  std::vector< std::vector<int> > totSIM_hit,totREC_hit,totASS_hit,totASS2_hit;
};


#endif
