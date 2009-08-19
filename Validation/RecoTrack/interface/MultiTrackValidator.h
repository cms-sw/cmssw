#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

/** \class MultiTrackValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2008/07/16 16:27:59 $
 *  $Revision: 1.45 $
 *  \author cerati
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

class MultiTrackValidator : public edm::EDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  MultiTrackValidator(const edm::ParameterSet& pset):MultiTrackValidatorBase(pset){
    dirName_ = pset.getParameter<std::string>("dirName");
    associatormap = pset.getParameter< edm::InputTag >("associatormap");
    UseAssociators = pset.getParameter< bool >("UseAssociators");
    tpSelector = TrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
					  pset.getParameter<double>("minRapidityTP"),
					  pset.getParameter<double>("maxRapidityTP"),
					  pset.getParameter<double>("tipTP"),
					  pset.getParameter<double>("lipTP"),
					  pset.getParameter<int>("minHitTP"),
					  pset.getParameter<bool>("signalOnlyTP"),
					  pset.getParameter<bool>("chargedOnlyTP"),
					  pset.getParameter<std::vector<int> >("pdgIdTP"));
    minPhi = pset.getParameter<double>("minPhi"); 
    maxPhi = pset.getParameter<double>("maxPhi");
    nintPhi = pset.getParameter<int>("nintPhi");
    
    if (!UseAssociators) {
      associators.clear();
      associators.push_back(associatormap.label());
    }
  }

  /// Destructor
  virtual ~MultiTrackValidator(){ }

  /// Method called before the event loop
  void beginRun(edm::Run const&, edm::EventSetup const&);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );
  /// Method called at the end of the event loop
  void endRun(edm::Run const&, edm::EventSetup const&);

 private:
  std::string dirName_;
  edm::InputTag associatormap;
  bool UseAssociators;
  double minPhi, maxPhi;
  int nintPhi;
  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      

  //1D
  std::vector<MonitorElement*> h_nchi2, h_nchi2_prob, h_losthits;

  //2D  
  std::vector<MonitorElement*> chi2_vs_nhits, etares_vs_eta;
  std::vector<MonitorElement*> h_ptshifteta;
  std::vector<MonitorElement*> ptres_vs_phi, chi2_vs_phi, nhits_vs_phi, phires_vs_phi;

  //Profile2D
  std::vector<MonitorElement*> ptmean_vs_eta_phi, phimean_vs_eta_phi;

  //assoc chi2
  std::vector<MonitorElement*> h_assochi2, h_assochi2_prob;

  //chi2 and # lost hits vs eta: to be used with doProfileX
  std::vector<MonitorElement*> chi2_vs_eta, nlosthits_vs_eta;
  std::vector<MonitorElement*> h_chi2meanh, h_losthits_eta;
  std::vector<MonitorElement*> h_hits_phi;  
  std::vector<MonitorElement*> h_chi2meanhitsh, h_chi2mean_vs_phi;

  //resolution of track params: to be used with fitslicesytool
  std::vector<MonitorElement*> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, cotThetares_vs_eta;
  std::vector<MonitorElement*> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, cotThetares_vs_pt;

  std::vector<MonitorElement*> h_dxyrmsh, h_ptrmsh, h_dzrmsh, h_phirmsh, h_cotThetarmsh;
  std::vector<MonitorElement*> h_dxyrmshPt, h_ptrmshPt, h_dzrmshPt, h_phirmshPt, h_cotThetarmshPt;
  std::vector<MonitorElement*> h_ptrmshPhi, h_phirmshPhi;
  //  std::vector<MonitorElement*> h_phimeanh,h_ptmeanhhi, h_phimeanhPhi;

  std::vector<MonitorElement*> h_dxymeanh, h_ptmeanh, h_dzmeanh, h_phimeanh, h_cotThetameanh;
  std::vector<MonitorElement*> h_dxymeanhPt, h_ptmeanhPt, h_dzmeanhPt, h_phimeanhPt, h_cotThetameanhPt;
  std::vector<MonitorElement*> h_ptmeanhPhi, h_phimeanhPhi;

  //pulls of track params vs eta: to be used with fitslicesytool
  std::vector<MonitorElement*> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<MonitorElement*> ptpull_vs_phi, phipull_vs_phi, thetapull_vs_phi;
  std::vector<MonitorElement*> h_dxypulleta, h_ptpulleta, h_dzpulleta, h_phipulleta, h_thetapulleta;
  std::vector<MonitorElement*> h_ptpullphi, h_phipullphi, h_thetapullphi;
  std::vector<MonitorElement*> h_dxypulletamean, h_ptpulletamean, h_dzpulletamean, h_phipulletamean, h_thetapulletamean;
  std::vector<MonitorElement*> h_ptpullphimean, h_phipullphimean, h_thetapullphimean;
};


#endif
