#ifndef MuonTrackValidatorBase_h
#define MuonTrackValidatorBase_h

/** \class MuonTrackValidatorBase
 *  Base class for analyzers that produces histograms to validate Muon Track Reconstruction performances
 *
 *  $Date: 2011/02/07 11:02:26 $
 *  $Revision: 1.5 $
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

#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

#include <iostream>
#include <sstream>
#include <string>
#include <TH1F.h>
#include <TH2F.h>

class MuonTrackValidatorBase {
 public:
  /// Constructor
  MuonTrackValidatorBase(const edm::ParameterSet& pset):
    label(pset.getParameter< std::vector<edm::InputTag> >("label")),
    usetracker(pset.getParameter<bool>("usetracker")),
    usemuon(pset.getParameter<bool>("usemuon")),
    bsSrc(pset.getParameter< edm::InputTag >("beamSpot")),
    label_tp_effic(pset.getParameter< edm::InputTag >("label_tp_effic")),
    label_tp_fake(pset.getParameter< edm::InputTag >("label_tp_fake")),
    associators(pset.getParameter< std::vector<std::string> >("associators")),
    out(pset.getParameter<std::string>("outputFile")),   
    parametersDefiner(pset.getParameter<std::string>("parametersDefiner")),
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
    minPhi(pset.getParameter<double>("minPhi")),
    maxPhi(pset.getParameter<double>("maxPhi")),
    nintPhi(pset.getParameter<int>("nintPhi")),
    minDxy(pset.getParameter<double>("minDxy")),
    maxDxy(pset.getParameter<double>("maxDxy")),
    nintDxy(pset.getParameter<int>("nintDxy")),
    minDz(pset.getParameter<double>("minDz")),
    maxDz(pset.getParameter<double>("maxDz")),
    nintDz(pset.getParameter<int>("nintDz")),
    minVertpos(pset.getParameter<double>("minVertpos")),
    maxVertpos(pset.getParameter<double>("maxVertpos")),
    nintVertpos(pset.getParameter<int>("nintVertpos")),
    minZpos(pset.getParameter<double>("minZpos")),
    maxZpos(pset.getParameter<double>("maxZpos")),
    nintZpos(pset.getParameter<int>("nintZpos")),
    useInvPt(pset.getParameter<bool>("useInvPt")),
    //
    ptRes_rangeMin(pset.getParameter<double>("ptRes_rangeMin")),
    ptRes_rangeMax(pset.getParameter<double>("ptRes_rangeMax")),
    phiRes_rangeMin(pset.getParameter<double>("phiRes_rangeMin")),
    phiRes_rangeMax(pset.getParameter<double>("phiRes_rangeMax")),
    cotThetaRes_rangeMin(pset.getParameter<double>("cotThetaRes_rangeMin")),
    cotThetaRes_rangeMax(pset.getParameter<double>("cotThetaRes_rangeMax")),
    dxyRes_rangeMin(pset.getParameter<double>("dxyRes_rangeMin")),
    dxyRes_rangeMax(pset.getParameter<double>("dxyRes_rangeMax")),
    dzRes_rangeMin(pset.getParameter<double>("dzRes_rangeMin")),
    dzRes_rangeMax(pset.getParameter<double>("dzRes_rangeMax")),
    ptRes_nbin(pset.getParameter<int>("ptRes_nbin")),
    cotThetaRes_nbin(pset.getParameter<int>("cotThetaRes_nbin")),
    phiRes_nbin(pset.getParameter<int>("phiRes_nbin")),
    dxyRes_nbin(pset.getParameter<int>("dxyRes_nbin")),
    dzRes_nbin(pset.getParameter<int>("dzRes_nbin")),
    ignoremissingtkcollection_(pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false)),
    useLogPt(pset.getUntrackedParameter<bool>("useLogPt",false))
    //
    {
      dbe_ = edm::Service<DQMStore>().operator->();
      if(useLogPt){
	maxpT=log10(maxpT);
	minpT=log10(minpT);
      }
    }
  
  /// Destructor
  virtual ~MuonTrackValidatorBase(){ }
  
  virtual void doProfileX(TH2 * th2, MonitorElement* me){
    if (th2->GetNbinsX()==me->getNbinsX()){
      TProfile * p1 = (TProfile*) th2->ProfileX();
      p1->Copy(*me->getTProfile());
      delete p1;
    } else {
      throw cms::Exception("MuonTrackValidator") << "Different number of bins!";
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

  void BinLogX(TH1*h)
  {
    
    TAxis *axis = h->GetXaxis();
    int bins = axis->GetNbins();
    
    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    float *new_bins = new float[bins + 1];
    
    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
      
    }
    axis->Set(bins, new_bins);
    delete[] new_bins;
  }

  void setUpVectors() {
    std::vector<double> etaintervalsv;
    std::vector<double> phiintervalsv;
    std::vector<double> pTintervalsv;
    std::vector<double> dxyintervalsv;
    std::vector<double> dzintervalsv;
    std::vector<double> vertposintervalsv;
    std::vector<double> zposintervalsv;
    std::vector<int>    totSIMveta,totASSveta,totASS2veta,totRECveta;
    std::vector<int>    totSIMvpT,totASSvpT,totASS2vpT,totRECvpT;
    std::vector<int>    totSIMv_hit,totASSv_hit,totASS2v_hit,totRECv_hit;
    std::vector<int>    totSIMv_phi,totASSv_phi,totASS2v_phi,totRECv_phi;
    std::vector<int>    totSIMv_dxy,totASSv_dxy,totASS2v_dxy,totRECv_dxy;
    std::vector<int>    totSIMv_dz,totASSv_dz,totASS2v_dz,totRECv_dz;
    std::vector<int>    totSIMv_vertpos,totASSv_vertpos,totSIMv_zpos,totASSv_zpos; 

    // for muon Validation
    std::vector<int>    totASSveta_Quality05, totASSveta_Quality075;
    std::vector<int>    totASSvpT_Quality05, totASSvpT_Quality075;
    std::vector<int>    totASSv_phi_Quality05, totASSv_phi_Quality075;

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
      //
      totASSveta_Quality05.push_back(0);
      totASSveta_Quality075.push_back(0);
    }   
    etaintervals.push_back(etaintervalsv);
    totSIMeta.push_back(totSIMveta);
    totASSeta.push_back(totASSveta);
    totASS2eta.push_back(totASS2veta);
    totRECeta.push_back(totRECveta);
    //
    totASSeta_Quality05.push_back(totASSveta_Quality05);
    totASSeta_Quality075.push_back(totASSveta_Quality075);
  
    double steppT = (maxpT-minpT)/nintpT;
    pTintervalsv.push_back(minpT);
    for (int k=1;k<nintpT+1;k++) {
      double d=0;
      if(useLogPt)d=pow(10,minpT+k*steppT);
      else d=minpT+k*steppT;
      pTintervalsv.push_back(d);
      totSIMvpT.push_back(0);
      totASSvpT.push_back(0);
      totASS2vpT.push_back(0);
      totRECvpT.push_back(0);
      //
      totASSvpT_Quality05.push_back(0);
      totASSvpT_Quality075.push_back(0);
    }
    pTintervals.push_back(pTintervalsv);
    totSIMpT.push_back(totSIMvpT);
    totASSpT.push_back(totASSvpT);
    totASS2pT.push_back(totASS2vpT);
    totRECpT.push_back(totRECvpT);
    //
    totASSpT_Quality05.push_back(totASSvpT_Quality05); 
    totASSpT_Quality075.push_back(totASSvpT_Quality075);

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

    double stepPhi = (maxPhi-minPhi)/nintPhi;
    phiintervalsv.push_back(minPhi);
    for (int k=1;k<nintPhi+1;k++) {
      double d=minPhi+k*stepPhi;
      phiintervalsv.push_back(d);
      totSIMv_phi.push_back(0);
      totASSv_phi.push_back(0);
      totASS2v_phi.push_back(0);
      totRECv_phi.push_back(0);
      //
      totASSv_phi_Quality05.push_back(0);
      totASSv_phi_Quality075.push_back(0);
    }
    phiintervals.push_back(phiintervalsv);
    totSIM_phi.push_back(totSIMv_phi);
    totASS_phi.push_back(totASSv_phi);
    totASS2_phi.push_back(totASS2v_phi);
    totREC_phi.push_back(totRECv_phi);
    //
    totASS_phi_Quality05.push_back(totASSv_phi_Quality05);
    totASS_phi_Quality075.push_back(totASSv_phi_Quality075);

    double stepDxy = (maxDxy-minDxy)/nintDxy;
    dxyintervalsv.push_back(minDxy);
    for (int k=1;k<nintDxy+1;k++) {
      double d=minDxy+k*stepDxy;
      dxyintervalsv.push_back(d);
      totSIMv_dxy.push_back(0);
      totASSv_dxy.push_back(0);
      totASS2v_dxy.push_back(0);
      totRECv_dxy.push_back(0);
    }
    dxyintervals.push_back(dxyintervalsv);
    totSIM_dxy.push_back(totSIMv_dxy);
    totASS_dxy.push_back(totASSv_dxy);
    totASS2_dxy.push_back(totASS2v_dxy);
    totREC_dxy.push_back(totRECv_dxy);


    double stepDz = (maxDz-minDz)/nintDz;
    dzintervalsv.push_back(minDz);
    for (int k=1;k<nintDz+1;k++) {
      double d=minDz+k*stepDz;
      dzintervalsv.push_back(d);
      totSIMv_dz.push_back(0);
      totASSv_dz.push_back(0);
      totASS2v_dz.push_back(0);
      totRECv_dz.push_back(0);
    }
    dzintervals.push_back(dzintervalsv);
    totSIM_dz.push_back(totSIMv_dz);
    totASS_dz.push_back(totASSv_dz);
    totASS2_dz.push_back(totASS2v_dz);
    totREC_dz.push_back(totRECv_dz);

    double stepVertpos = (maxVertpos-minVertpos)/nintVertpos;
    vertposintervalsv.push_back(minVertpos);
    for (int k=1;k<nintVertpos+1;k++) {
      double d=minVertpos+k*stepVertpos;
      vertposintervalsv.push_back(d);
      totSIMv_vertpos.push_back(0);
      totASSv_vertpos.push_back(0);
    }
    vertposintervals.push_back(vertposintervalsv);
    totSIM_vertpos.push_back(totSIMv_vertpos);
    totASS_vertpos.push_back(totASSv_vertpos);
      
    double stepZpos = (maxZpos-minZpos)/nintZpos;
    zposintervalsv.push_back(minZpos);
    for (int k=1;k<nintZpos+1;k++) {
      double d=minZpos+k*stepZpos;
      zposintervalsv.push_back(d);
      totSIMv_zpos.push_back(0);
      totASSv_zpos.push_back(0);
    }
    zposintervals.push_back(zposintervalsv);
    totSIM_zpos.push_back(totSIMv_zpos);
    totASS_zpos.push_back(totASSv_zpos);

  }

 protected:

  DQMStore* dbe_;

  std::vector<edm::InputTag> label;
  bool usetracker;
  bool usemuon;
  edm::InputTag bsSrc;
  edm::InputTag label_tp_effic;
  edm::InputTag label_tp_fake;
  std::vector<std::string> associators;
  std::string out;
  std::string parametersDefiner;
       
  double  min, max;
  int nint;
  bool useFabs;
  double minpT, maxpT;
  int nintpT;
  double minHit, maxHit;
  int nintHit;
  double minPhi, maxPhi;
  int nintPhi;
  double minDxy, maxDxy;
  int nintDxy;
  double minDz, maxDz;
  int nintDz;
  double minVertpos, maxVertpos;
  int nintVertpos;
  double minZpos, maxZpos;
  int nintZpos;
  bool useInvPt;
  //
  double ptRes_rangeMin,ptRes_rangeMax,
    phiRes_rangeMin,phiRes_rangeMax, cotThetaRes_rangeMin,cotThetaRes_rangeMax,    
    dxyRes_rangeMin,dxyRes_rangeMax, dzRes_rangeMin,dzRes_rangeMax;
  int ptRes_nbin, cotThetaRes_nbin, phiRes_nbin, dxyRes_nbin, dzRes_nbin;
  bool ignoremissingtkcollection_;
  bool useLogPt;

  edm::ESHandle<MagneticField> theMF;
  std::vector<const TrackAssociatorBase*> associator;

  //sim
  std::vector<MonitorElement*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;

  //1D
  std::vector<MonitorElement*> h_tracks, h_fakes, h_hits, h_charge;
  std::vector<MonitorElement*> h_recoeta, h_assoceta, h_assoc2eta, h_simuleta;
  std::vector<MonitorElement*> h_recopT, h_assocpT, h_assoc2pT, h_simulpT;
  std::vector<MonitorElement*> h_recohit, h_assochit, h_assoc2hit, h_simulhit;
  std::vector<MonitorElement*> h_recophi, h_assocphi, h_assoc2phi, h_simulphi;
  std::vector<MonitorElement*> h_recodxy, h_assocdxy, h_assoc2dxy, h_simuldxy;
  std::vector<MonitorElement*> h_recodz, h_assocdz, h_assoc2dz, h_simuldz;
  std::vector<MonitorElement*> h_assocvertpos, h_simulvertpos, h_assoczpos, h_simulzpos;
  std::vector<MonitorElement*> h_pt, h_eta, h_pullTheta,h_pullPhi,h_pullDxy,h_pullDz,h_pullQoverp;

  std::vector<MonitorElement*> h_assoceta_Quality05, h_assoceta_Quality075;
  std::vector<MonitorElement*> h_assocpT_Quality05, h_assocpT_Quality075;
  std::vector<MonitorElement*> h_assocphi_Quality05, h_assocphi_Quality075;


  //2D  
  std::vector<MonitorElement*> nrec_vs_nsim;
  std::vector<MonitorElement*> nrecHit_vs_nsimHit_sim2rec;
  std::vector<MonitorElement*> nrecHit_vs_nsimHit_rec2sim;

  //assoc hits
  std::vector<MonitorElement*> h_assocFraction, h_assocSharedHit;

  //#hit vs eta: to be used with doProfileX
  std::vector<MonitorElement*> nhits_vs_eta, 
    nDThits_vs_eta,nCSChits_vs_eta,nRPChits_vs_eta,nGEMhits_vs_eta;

  std::vector<MonitorElement*> h_hits_eta,
    h_DThits_eta,h_CSChits_eta,h_RPChits_eta,h_GEMhits_eta;
    

  std::vector< std::vector<double> > etaintervals;
  std::vector< std::vector<double> > pTintervals;
  std::vector< std::vector<double> > phiintervals;
  std::vector< std::vector<double> > dxyintervals;
  std::vector< std::vector<double> > dzintervals;
  std::vector< std::vector<double> > vertposintervals;
  std::vector< std::vector<double> > zposintervals;
  std::vector< std::vector<int> > totSIMeta,totRECeta,totASSeta,totASS2eta;
  std::vector< std::vector<int> > totSIMpT,totRECpT,totASSpT,totASS2pT;
  std::vector< std::vector<int> > totSIM_hit,totREC_hit,totASS_hit,totASS2_hit;
  std::vector< std::vector<int> > totSIM_phi,totREC_phi,totASS_phi,totASS2_phi;
  std::vector< std::vector<int> > totSIM_dxy,totREC_dxy,totASS_dxy,totASS2_dxy;
  std::vector< std::vector<int> > totSIM_dz,totREC_dz,totASS_dz,totASS2_dz;
  std::vector< std::vector<int> > totSIM_vertpos,totASS_vertpos,totSIM_zpos,totASS_zpos;

  // for muon Validation (SimToReco distributions for Quality > 0.5, 0.75)
  std::vector<MonitorElement*> h_PurityVsQuality;
  std::vector< std::vector<int> > totASSeta_Quality05,totASSeta_Quality075;
  std::vector< std::vector<int> > totASSpT_Quality05, totASSpT_Quality075;
  std::vector< std::vector<int> > totASS_phi_Quality05, totASS_phi_Quality075;

};


#endif
