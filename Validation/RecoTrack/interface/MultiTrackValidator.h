#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

/** \class MultiTrackValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2008/02/21 09:36:20 $
 *  $Revision: 1.40 $
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
  }

  /// Destructor
  virtual ~MultiTrackValidator(){ }

  /// Method called before the event loop
  void beginJob( const edm::EventSetup &);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );
  /// Method called at the end of the event loop
  void endJob();

 private:
  std::string dirName_;
  //1D
  std::vector<MonitorElement*> h_nchi2, h_nchi2_prob, h_losthits;

  //2D  
  std::vector<MonitorElement*> chi2_vs_nhits, etares_vs_eta;
  std::vector<MonitorElement*> h_ptshifteta;

  //assoc chi2
  std::vector<MonitorElement*> h_assochi2, h_assochi2_prob;

  //chi2 and # lost hits vs eta: to be used with doProfileX
  std::vector<MonitorElement*> chi2_vs_eta, nlosthits_vs_eta;
  std::vector<MonitorElement*> h_chi2meanh, h_losthits_eta;

  //resolution of track params: to be used with fitslicesytool
  std::vector<MonitorElement*> dxyres_vs_eta, ptres_vs_eta, dzres_vs_eta, phires_vs_eta, cotThetares_vs_eta;
  std::vector<MonitorElement*> dxyres_vs_pt, ptres_vs_pt, dzres_vs_pt, phires_vs_pt, cotThetares_vs_pt;
  std::vector<MonitorElement*> h_dxyrmsh, h_ptrmsh, h_dzrmsh, h_phirmsh, h_cotThetarmsh;
  std::vector<MonitorElement*> h_dxyrmshPt, h_ptrmshPt, h_dzrmshPt, h_phirmshPt, h_cotThetarmshPt;
  //pulls of track params vs eta: to be used with fitslicesytool
  std::vector<MonitorElement*> dxypull_vs_eta, ptpull_vs_eta, dzpull_vs_eta, phipull_vs_eta, thetapull_vs_eta;
  std::vector<MonitorElement*> h_dxypulleta, h_ptpulleta, h_dzpulleta, h_phipulleta, h_thetapulleta;
};


#endif
