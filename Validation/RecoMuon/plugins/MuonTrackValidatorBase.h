#ifndef MuonTrackValidatorBase_h
#define MuonTrackValidatorBase_h

/** \class MuonTrackValidatorBase
* Base class for analyzers that produces histograms to validate Muon Track Reconstruction performances
*
*/

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

#include <iostream>
#include <sstream>
#include <string>
#include <TH1F.h>
#include <TH2F.h>


class DQMStore;
class MuonTrackValidatorBase {
 public:
  /// Constructor
  MuonTrackValidatorBase(const edm::ParameterSet& pset, edm::ConsumesCollector iC) : MuonTrackValidatorBase(pset)
    {
      bsSrc_Token = iC.consumes<reco::BeamSpot>(bsSrc);
      tp_effic_Token = iC.consumes<TrackingParticleCollection>(label_tp_effic);
      tp_fake_Token = iC.consumes<TrackingParticleCollection>(label_tp_fake);
      pileupinfo_Token = iC.consumes<std::vector<PileupSummaryInfo> >(label_pileupinfo);
      for (unsigned int www=0;www<label.size();www++){
	track_Collection_Token[www] = iC.consumes<edm::View<reco::Track> >(label[www]);
      }
    }

  MuonTrackValidatorBase(const edm::ParameterSet& pset):
    label(pset.getParameter< std::vector<edm::InputTag> >("label")),
    bsSrc(pset.getParameter< edm::InputTag >("beamSpot")),
    label_tp_effic(pset.getParameter< edm::InputTag >("label_tp_effic")),
    label_tp_fake(pset.getParameter< edm::InputTag >("label_tp_fake")),
    label_pileupinfo(pset.getParameter< edm::InputTag >("label_pileupinfo")),
    associators(pset.getParameter< std::vector<std::string> >("associators")),
    out(pset.getParameter<std::string>("outputFile")),
    parametersDefiner(pset.getParameter<std::string>("parametersDefiner")),
    muonHistoParameters(pset.getParameter<edm::ParameterSet>("muonHistoParameters")),
    ignoremissingtkcollection_(pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false))

    {
      minEta = muonHistoParameters.getParameter<double>("minEta");
      maxEta = muonHistoParameters.getParameter<double>("maxEta");
      nintEta = muonHistoParameters.getParameter<int>("nintEta");
      useFabsEta = muonHistoParameters.getParameter<bool>("useFabsEta");
      minPt = muonHistoParameters.getParameter<double>("minPt");
      maxPt = muonHistoParameters.getParameter<double>("maxPt");
      nintPt = muonHistoParameters.getParameter<int>("nintPt");
      useLogPt = muonHistoParameters.getUntrackedParameter<bool>("useLogPt",false);
      useInvPt = muonHistoParameters.getParameter<bool>("useInvPt");
      minNHit = muonHistoParameters.getParameter<double>("minNHit");
      maxNHit = muonHistoParameters.getParameter<double>("maxNHit");
      nintNHit = muonHistoParameters.getParameter<int>("nintNHit");
      //
      minDTHit = muonHistoParameters.getParameter<double>("minDTHit");
      maxDTHit = muonHistoParameters.getParameter<double>("maxDTHit");
      nintDTHit = muonHistoParameters.getParameter<int>("nintDTHit");
      //
      minCSCHit = muonHistoParameters.getParameter<double>("minCSCHit");
      maxCSCHit = muonHistoParameters.getParameter<double>("maxCSCHit");
      nintCSCHit = muonHistoParameters.getParameter<int>("nintCSCHit");
      //
      minRPCHit = muonHistoParameters.getParameter<double>("minRPCHit");
      maxRPCHit = muonHistoParameters.getParameter<double>("maxRPCHit");
      nintRPCHit = muonHistoParameters.getParameter<int>("nintRPCHit");
      //
      minLayers = muonHistoParameters.getParameter<double>("minLayers");
      maxLayers = muonHistoParameters.getParameter<double>("maxLayers");
      nintLayers = muonHistoParameters.getParameter<int>("nintLayers");
      minPixels = muonHistoParameters.getParameter<double>("minPixels");
      maxPixels = muonHistoParameters.getParameter<double>("maxPixels");
      nintPixels = muonHistoParameters.getParameter<int>("nintPixels");
      minPhi = muonHistoParameters.getParameter<double>("minPhi");
      maxPhi = muonHistoParameters.getParameter<double>("maxPhi");
      nintPhi = muonHistoParameters.getParameter<int>("nintPhi");
      minDxy = muonHistoParameters.getParameter<double>("minDxy");
      maxDxy = muonHistoParameters.getParameter<double>("maxDxy");
      nintDxy = muonHistoParameters.getParameter<int>("nintDxy");
      minDz = muonHistoParameters.getParameter<double>("minDz");
      maxDz = muonHistoParameters.getParameter<double>("maxDz");
      nintDz = muonHistoParameters.getParameter<int>("nintDz");
      minRpos = muonHistoParameters.getParameter<double>("minRpos");
      maxRpos = muonHistoParameters.getParameter<double>("maxRpos");
      nintRpos = muonHistoParameters.getParameter<int>("nintRpos");
      minZpos = muonHistoParameters.getParameter<double>("minZpos");
      maxZpos = muonHistoParameters.getParameter<double>("maxZpos");
      nintZpos = muonHistoParameters.getParameter<int>("nintZpos");
      minPU = muonHistoParameters.getParameter<double>("minPU");
      maxPU = muonHistoParameters.getParameter<double>("maxPU");
      nintPU = muonHistoParameters.getParameter<int>("nintPU");
      //
      ptRes_rangeMin = muonHistoParameters.getParameter<double>("ptRes_rangeMin");
      ptRes_rangeMax = muonHistoParameters.getParameter<double>("ptRes_rangeMax");
      ptRes_nbin = muonHistoParameters.getParameter<int>("ptRes_nbin");
      etaRes_rangeMin = muonHistoParameters.getParameter<double>("etaRes_rangeMin");
      etaRes_rangeMax = muonHistoParameters.getParameter<double>("etaRes_rangeMax");
      etaRes_nbin = muonHistoParameters.getParameter<int>("etaRes_nbin");
      phiRes_rangeMin = muonHistoParameters.getParameter<double>("phiRes_rangeMin");
      phiRes_rangeMax = muonHistoParameters.getParameter<double>("phiRes_rangeMax");
      phiRes_nbin = muonHistoParameters.getParameter<int>("phiRes_nbin");
      cotThetaRes_rangeMin = muonHistoParameters.getParameter<double>("cotThetaRes_rangeMin");
      cotThetaRes_rangeMax = muonHistoParameters.getParameter<double>("cotThetaRes_rangeMax");
      cotThetaRes_nbin = muonHistoParameters.getParameter<int>("cotThetaRes_nbin");
      dxyRes_rangeMin = muonHistoParameters.getParameter<double>("dxyRes_rangeMin");
      dxyRes_rangeMax = muonHistoParameters.getParameter<double>("dxyRes_rangeMax");
      dxyRes_nbin = muonHistoParameters.getParameter<int>("dxyRes_nbin");
      dzRes_rangeMin = muonHistoParameters.getParameter<double>("dzRes_rangeMin");
      dzRes_rangeMax = muonHistoParameters.getParameter<double>("dzRes_rangeMax");
      dzRes_nbin = muonHistoParameters.getParameter<int>("dzRes_nbin");
      //
      usetracker = muonHistoParameters.getParameter<bool>("usetracker");
      usemuon = muonHistoParameters.getParameter<bool>("usemuon");
      do_TRKhitsPlots = muonHistoParameters.getParameter<bool>("do_TRKhitsPlots");
      do_MUOhitsPlots = muonHistoParameters.getParameter<bool>("do_MUOhitsPlots");
      
      if (useLogPt) {
        minPt=log10(std::max(0.01,minPt));
	maxPt=log10(maxPt);
      }
    }
  
  /// Destructor
  virtual ~MuonTrackValidatorBase() noexcept(false) { }
 
  template<typename T> void fillPlotNoFlow (MonitorElement* h, T val) {
    h->Fill(std::min(std::max(val,((T) h->getTH1()->GetXaxis()->GetXmin())),((T) h->getTH1()->GetXaxis()->GetXmax())));
  }
  
  void doProfileX(TH2 * th2, MonitorElement* me){
    if (th2->GetNbinsX()==me->getNbinsX()){
      TProfile * p1 = (TProfile*) th2->ProfileX();
      p1->Copy(*me->getTProfile());
      delete p1;
    } else {
      throw cms::Exception("MuonTrackValidator") << "Different number of bins!";
    }
  }

  void doProfileX(MonitorElement * th2m, MonitorElement* me) {
    doProfileX(th2m->getTH2F(), me);
  }

  //  virtual double getEta(double eta) {
  double getEta(double eta) {
    if (useFabsEta) return fabs(eta);
    else return eta;
  }

  //  virtual double getPt(double pt) {
  double getPt(double pt) {
    if (useInvPt && pt!=0) return 1/pt;
    else return pt;
  }
  
  void BinLogX(TH1*h) {
    
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

 protected:

  std::vector<edm::InputTag> label;
  edm::InputTag bsSrc;
  edm::InputTag label_tp_effic;
  edm::InputTag label_tp_fake;
  edm::InputTag label_pileupinfo;
  std::vector<std::string> associators;
  std::string out;
  std::string parametersDefiner;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track> > > track_Collection_Token;
  edm::EDGetTokenT<reco::BeamSpot> bsSrc_Token;
  edm::EDGetTokenT<TrackingParticleCollection> tp_effic_Token;
  edm::EDGetTokenT<TrackingParticleCollection> tp_fake_Token;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > pileupinfo_Token;
  edm::ESHandle<MagneticField> theMF;

  edm::ParameterSet muonHistoParameters;

  double minEta, maxEta;  int nintEta;  bool useFabsEta;
  double minPt, maxPt;  int nintPt;  bool useLogPt;  bool useInvPt; 
  double minNHit, maxNHit;  int nintNHit;
  double minDTHit, maxDTHit; int nintDTHit;
  double minCSCHit, maxCSCHit;  int nintCSCHit;
  double minRPCHit, maxRPCHit;  int nintRPCHit;
  double minLayers, maxLayers;  int nintLayers;
  double minPixels, maxPixels;  int nintPixels;
  double minPhi, maxPhi;  int nintPhi;
  double minDxy, maxDxy;  int nintDxy;
  double minDz, maxDz;  int nintDz;
  double minRpos, maxRpos;  int nintRpos;
  double minZpos, maxZpos;  int nintZpos;
  double minPU, maxPU;  int nintPU;
  //
  double ptRes_rangeMin,ptRes_rangeMax; int ptRes_nbin;
  double etaRes_rangeMin,etaRes_rangeMax; int etaRes_nbin;
  double phiRes_rangeMin,phiRes_rangeMax; int phiRes_nbin;
  double cotThetaRes_rangeMin,cotThetaRes_rangeMax; int cotThetaRes_nbin;
  double dxyRes_rangeMin,dxyRes_rangeMax; int dxyRes_nbin;
  double dzRes_rangeMin,dzRes_rangeMax; int dzRes_nbin;
      
  bool usetracker, usemuon;
  bool do_TRKhitsPlots, do_MUOhitsPlots;
  bool ignoremissingtkcollection_;

  //1D
  std::vector<MonitorElement*> h_tracks, h_fakes, h_nhits, h_charge;
  std::vector<MonitorElement*> h_recoeta, h_assoceta, h_assoc2eta, h_simuleta, h_misideta;
  std::vector<MonitorElement*> h_recopT, h_assocpT, h_assoc2pT, h_simulpT, h_misidpT;
  std::vector<MonitorElement*> h_recohit, h_assochit, h_assoc2hit, h_simulhit, h_misidhit;
  std::vector<MonitorElement*> h_recophi, h_assocphi, h_assoc2phi, h_simulphi, h_misidphi;
  std::vector<MonitorElement*> h_recodxy, h_assocdxy, h_assoc2dxy, h_simuldxy, h_misiddxy;
  std::vector<MonitorElement*> h_recodz, h_assocdz, h_assoc2dz, h_simuldz, h_misiddz;
  std::vector<MonitorElement*> h_recopu, h_assocpu, h_assoc2pu, h_simulpu, h_misidpu;

  std::vector<MonitorElement*> h_assocRpos, h_simulRpos, h_assocZpos, h_simulZpos;
  std::vector<MonitorElement*> h_etaRes;

  std::vector<MonitorElement*> h_nchi2, h_nchi2_prob, h_losthits;
  std::vector<MonitorElement*> h_nmisslayers_inner,h_nmisslayers_outer,h_nlosthits;
  std::vector<MonitorElement*> h_assochi2, h_assochi2_prob;
  std::vector<MonitorElement*> h_assocFraction, h_assocSharedHit;
  
  //2D
  std::vector<MonitorElement*> nRecHits_vs_nSimHits;
  std::vector<MonitorElement*> h_PurityVsQuality;
  std::vector<MonitorElement*> chi2_vs_nhits, etares_vs_eta;
  std::vector<MonitorElement*> ptres_vs_phi, chi2_vs_phi, nhits_vs_phi, phires_vs_phi;

  std::vector<MonitorElement*> nhits_vs_eta,nDThits_vs_eta,nCSChits_vs_eta,nRPChits_vs_eta,nGEMhits_vs_eta,nME0hits_vs_eta;
  std::vector<MonitorElement*> chi2_vs_eta, nlosthits_vs_eta;
  std::vector<MonitorElement*> nTRK_LayersWithMeas_vs_eta,nPixel_LayersWithMeas_vs_eta;

  std::vector<MonitorElement*> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, thetaCotres_vs_eta;
  std::vector<MonitorElement*> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, thetaCotres_vs_pt;

  std::vector<MonitorElement*> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<MonitorElement*> ptpull_vs_phi, phipull_vs_phi, thetapull_vs_phi;
  std::vector<MonitorElement*> h_dxypulleta, h_ptpulleta, h_dzpulleta, h_phipulleta, h_thetapulleta;
  std::vector<MonitorElement*> h_ptpullphi, h_phipullphi, h_thetapullphi;
  std::vector<MonitorElement*> h_ptpull, h_qoverppull, h_thetapull, h_phipull, h_dxypull, h_dzpull;

};


#endif
