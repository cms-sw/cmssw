#ifndef Validation_EcalClusters_EgammaBasicClusters_h
#define Validation_EcalClusters_EgammaBasicClusters_h

/**\class EgammaBasicClusters

   Description: SVSuite Basic Cluster Validation

   Implementation:
   \\\author: Michael A. Balazs, Nov 2006
*/
//
//

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "HistSpec.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

class EgammaBasicClusters : public DQMEDAnalyzer {
public:
  explicit EgammaBasicClusters(const edm::ParameterSet &);
  ~EgammaBasicClusters() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  const bool enableEndcaps_;
  const edm::EDGetTokenT<reco::BasicClusterCollection> barrelBasicClusterCollection_;
  const edm::EDGetTokenT<reco::BasicClusterCollection> endcapBasicClusterCollection_;

  const HistSpec hsSize_;
  const HistSpec hsNumRecHits_;
  const HistSpec hsET_;
  const HistSpec hsEta_;
  const HistSpec hsPhi_;
  const HistSpec hsR_;

  MonitorElement *hist_EB_BC_Size_;
  MonitorElement *hist_EE_BC_Size_;
  MonitorElement *hist_EB_BC_NumRecHits_;
  MonitorElement *hist_EE_BC_NumRecHits_;
  MonitorElement *hist_EB_BC_ET_;
  MonitorElement *hist_EE_BC_ET_;
  MonitorElement *hist_EB_BC_Eta_;
  MonitorElement *hist_EE_BC_Eta_;
  MonitorElement *hist_EB_BC_Phi_;
  MonitorElement *hist_EE_BC_Phi_;
  MonitorElement *hist_EB_BC_ET_vs_Eta_;
  MonitorElement *hist_EB_BC_ET_vs_Phi_;
  MonitorElement *hist_EE_BC_ET_vs_Eta_;
  MonitorElement *hist_EE_BC_ET_vs_Phi_;
  MonitorElement *hist_EE_BC_ET_vs_R_;
};

#endif
