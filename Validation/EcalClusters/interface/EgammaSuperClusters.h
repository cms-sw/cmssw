#ifndef Validation_EcalClusters_EgammaSuperClusters_h
#define Validation_EcalClusters_EgammaSuperClusters_h

/**\class EgammaSuperClusters

 Description: SVSuite Super Cluster Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
//

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "HistSpec.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

class EgammaSuperClusters : public DQMEDAnalyzer {
public:
  explicit EgammaSuperClusters(const edm::ParameterSet &);
  ~EgammaSuperClusters() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  const bool enableEndcaps_;
  // mc truth
  const edm::EDGetTokenT<edm::HepMCProduct> MCTruthCollectionToken_;

  // barrel clusters
  const edm::EDGetTokenT<reco::SuperClusterCollection> barrelRawSuperClusterCollectionToken_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> barrelCorSuperClusterCollectionToken_;

  // endcap clusters
  const edm::EDGetTokenT<reco::SuperClusterCollection> endcapRawSuperClusterCollectionToken_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> endcapPreSuperClusterCollectionToken_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> endcapCorSuperClusterCollectionToken_;

  // collections of hits
  const edm::EDGetTokenT<EcalRecHitCollection> barrelRecHitCollectionToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapRecHitCollectionToken_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  const HistSpec hsSize_;
  const HistSpec hsNumBC_;
  const HistSpec hsET_;
  const HistSpec hsEta_;
  const HistSpec hsPhi_;
  const HistSpec hsS1toS9_;
  const HistSpec hsS25toE_;
  const HistSpec hsEoverTruth_;
  const HistSpec hsdeltaR_;
  const HistSpec hsphiWidth_;
  const HistSpec hsetaWidth_;
  const HistSpec hspreshowerE_;
  const HistSpec hsR_;

  MonitorElement *hist_EB_RawSC_Size_;
  MonitorElement *hist_EE_RawSC_Size_;
  MonitorElement *hist_EB_CorSC_Size_;
  MonitorElement *hist_EE_CorSC_Size_;
  MonitorElement *hist_EE_PreSC_Size_;
  MonitorElement *hist_EB_RawSC_NumBC_;
  MonitorElement *hist_EE_RawSC_NumBC_;
  MonitorElement *hist_EB_CorSC_NumBC_;
  MonitorElement *hist_EE_CorSC_NumBC_;
  MonitorElement *hist_EE_PreSC_NumBC_;
  MonitorElement *hist_EB_RawSC_ET_;
  MonitorElement *hist_EE_RawSC_ET_;
  MonitorElement *hist_EB_CorSC_ET_;
  MonitorElement *hist_EE_CorSC_ET_;
  MonitorElement *hist_EE_PreSC_ET_;
  MonitorElement *hist_EB_RawSC_Eta_;
  MonitorElement *hist_EE_RawSC_Eta_;
  MonitorElement *hist_EB_CorSC_Eta_;
  MonitorElement *hist_EE_CorSC_Eta_;
  MonitorElement *hist_EE_PreSC_Eta_;
  MonitorElement *hist_EB_RawSC_Phi_;
  MonitorElement *hist_EE_RawSC_Phi_;
  MonitorElement *hist_EB_CorSC_Phi_;
  MonitorElement *hist_EE_CorSC_Phi_;
  MonitorElement *hist_EE_PreSC_Phi_;
  MonitorElement *hist_EB_RawSC_S1toS9_;
  MonitorElement *hist_EE_RawSC_S1toS9_;
  MonitorElement *hist_EB_CorSC_S1toS9_;
  MonitorElement *hist_EE_CorSC_S1toS9_;
  MonitorElement *hist_EE_PreSC_S1toS9_;
  MonitorElement *hist_EB_RawSC_S25toE_;
  MonitorElement *hist_EE_RawSC_S25toE_;
  MonitorElement *hist_EB_CorSC_S25toE_;
  MonitorElement *hist_EE_CorSC_S25toE_;
  MonitorElement *hist_EE_PreSC_S25toE_;
  MonitorElement *hist_EB_RawSC_EoverTruth_;
  MonitorElement *hist_EE_RawSC_EoverTruth_;
  MonitorElement *hist_EB_CorSC_EoverTruth_;
  MonitorElement *hist_EE_CorSC_EoverTruth_;
  MonitorElement *hist_EE_PreSC_EoverTruth_;
  MonitorElement *hist_EB_RawSC_deltaR_;
  MonitorElement *hist_EE_RawSC_deltaR_;
  MonitorElement *hist_EB_CorSC_deltaR_;
  MonitorElement *hist_EE_CorSC_deltaR_;
  MonitorElement *hist_EE_PreSC_deltaR_;
  MonitorElement *hist_EE_PreSC_preshowerE_;
  MonitorElement *hist_EE_CorSC_preshowerE_;
  MonitorElement *hist_EE_CorSC_phiWidth_;
  MonitorElement *hist_EB_CorSC_phiWidth_;
  MonitorElement *hist_EE_CorSC_etaWidth_;
  MonitorElement *hist_EB_CorSC_etaWidth_;
  MonitorElement *hist_EB_CorSC_ET_vs_Eta_;
  MonitorElement *hist_EB_CorSC_ET_vs_Phi_;
  MonitorElement *hist_EE_CorSC_ET_vs_Eta_;
  MonitorElement *hist_EE_CorSC_ET_vs_Phi_;
  MonitorElement *hist_EE_CorSC_ET_vs_R_;

  void closestMCParticle(HepMC::GenEvent const *, reco::SuperCluster const &, double &, double &) const;

  float ecalEta(float, float, float) const;
};

#endif
