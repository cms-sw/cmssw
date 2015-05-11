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

#include <string>

class MonitorElement;

class EgammaBasicClusters : public DQMEDAnalyzer
{
 public:
  explicit EgammaBasicClusters( const edm::ParameterSet& );
  ~EgammaBasicClusters();

  void analyze( const edm::Event&, const edm::EventSetup& ) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

 private:
  edm::EDGetTokenT<reco::BasicClusterCollection> barrelBasicClusterCollection_;
  edm::EDGetTokenT<reco::BasicClusterCollection> endcapBasicClusterCollection_;
 
  HistSpec hsSize_;
  HistSpec hsNumRecHits_;
  HistSpec hsET_;
  HistSpec hsEta_;
  HistSpec hsPhi_;
  HistSpec hsR_;

  MonitorElement* hist_EB_BC_Size_;
  MonitorElement* hist_EE_BC_Size_;
  MonitorElement* hist_EB_BC_NumRecHits_;
  MonitorElement* hist_EE_BC_NumRecHits_;
  MonitorElement* hist_EB_BC_ET_;
  MonitorElement* hist_EE_BC_ET_;
  MonitorElement* hist_EB_BC_Eta_;
  MonitorElement* hist_EE_BC_Eta_;
  MonitorElement* hist_EB_BC_Phi_;
  MonitorElement* hist_EE_BC_Phi_;
  MonitorElement* hist_EB_BC_ET_vs_Eta_;
  MonitorElement* hist_EB_BC_ET_vs_Phi_;
  MonitorElement* hist_EE_BC_ET_vs_Eta_;
  MonitorElement* hist_EE_BC_ET_vs_Phi_;
  MonitorElement* hist_EE_BC_ET_vs_R_;
};

#endif
