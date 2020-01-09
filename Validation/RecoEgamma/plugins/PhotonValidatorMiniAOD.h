#ifndef PhotonValidatorMiniAOD_H
#define PhotonValidatorMiniAOD_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//
#include <map>
#include <vector>
#include <memory>
/** \class PhotonValidatorMiniAOD
 **
 **
 **  $Id: PhotonValidatorMiniAOD
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

// forward declarations
namespace edm {
  class HepMCProduct;
}
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;

class PhotonValidatorMiniAOD : public DQMEDAnalyzer {
public:
  //
  explicit PhotonValidatorMiniAOD(const edm::ParameterSet&);
  ~PhotonValidatorMiniAOD() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //  virtual void beginJob();
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  std::string outputFileName_;
  edm::EDGetTokenT<edm::View<pat::Photon> > photonToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genpartToken_;

  std::string fName_;

  //  edm::ESHandle<MagneticField> theMF_;
  edm::ParameterSet parameters_;
  int verbosity_;

  MonitorElement* h_scEta_[2];
  MonitorElement* h_scPhi_[2];

  MonitorElement* h_r9_[3][3];
  MonitorElement* h_full5x5_r9_[3][3];
  MonitorElement* h_sigmaIetaIeta_[3][3];
  MonitorElement* h_full5x5_sigmaIetaIeta_[3][3];
  MonitorElement* h_r1_[3][3];
  MonitorElement* h_r2_[3][3];
  MonitorElement* h_hOverE_[3][3];
  MonitorElement* h_newhOverE_[3][3];
  MonitorElement* h_ecalRecHitSumEtConeDR04_[3][3];
  MonitorElement* h_hcalTowerSumEtConeDR04_[3][3];
  MonitorElement* h_hcalTowerBcSumEtConeDR04_[3][3];
  MonitorElement* h_isoTrkSolidConeDR04_[3][3];
  MonitorElement* h_nTrkSolidConeDR04_[3][3];

  MonitorElement* h_phoE_[2][3];
  MonitorElement* h_phoEt_[2][3];
  MonitorElement* h_phoERes_[3][3];
  MonitorElement* h_phoSigmaEoE_[3][3];

  // Information from Particle Flow
  // Isolation
  MonitorElement* h_chHadIso_[3];
  MonitorElement* h_nHadIso_[3];
  MonitorElement* h_phoIso_[3];

  class sortPhotons {
  public:
    bool operator()(const pat::PhotonRef& lhs, const pat::PhotonRef& rhs) { return lhs->et() > rhs->et(); }
  };
};

#endif
