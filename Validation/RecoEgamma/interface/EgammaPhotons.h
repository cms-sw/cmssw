#ifndef Validation_RecoEgamma_EgammaPhotons_h
#define Validation_RecoEgamma_EgammaPhotons_h

/**\class EgammaPhotons

 Description: SVSuite Photon Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaPhotons.h,v 1.5 2006/12/06 16:39:18 mabalazs Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

class EgammaPhotons : public edm::EDAnalyzer
{
  public:
    explicit EgammaPhotons( const edm::ParameterSet& );
    ~EgammaPhotons();

    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    virtual void beginJob(edm::EventSetup const&);
    virtual void endJob();

  private:
    std::string outputFile_;
    std::string CMSSW_Version_;

    bool verboseDBE_;
    DaqMonitorBEInterface* dbe_;

    edm::InputTag MCTruthCollection_;
    edm::InputTag PhotonCollection_;

    MonitorElement* hist_Photon_Size_;

    double hist_min_Size_;
    double hist_max_Size_;
    int    hist_bins_Size_;

    MonitorElement* hist_Photon_Barrel_ET_;
    MonitorElement* hist_Photon_Endcap_ET_;

    double hist_min_ET_;
    double hist_max_ET_;
    int    hist_bins_ET_;

    MonitorElement* hist_Photon_Barrel_Eta_;
    MonitorElement* hist_Photon_Endcap_Eta_;

    double hist_min_Eta_;
    double hist_max_Eta_;
    int    hist_bins_Eta_;

    MonitorElement* hist_Photon_Barrel_Phi_;
    MonitorElement* hist_Photon_Endcap_Phi_;

    double hist_min_Phi_;
    double hist_max_Phi_;
    int    hist_bins_Phi_;

    MonitorElement* hist_Photon_Barrel_EToverTruth_;
    MonitorElement* hist_Photon_Endcap_EToverTruth_;

    double hist_min_EToverTruth_;
    double hist_max_EToverTruth_;
    int    hist_bins_EToverTruth_;

    MonitorElement* hist_Photon_Barrel_deltaEta_;
    MonitorElement* hist_Photon_Endcap_deltaEta_;

    double hist_min_deltaEta_;
    double hist_max_deltaEta_;
    int    hist_bins_deltaEta_;

    MonitorElement* hist_Photon_Barrel_deltaPhi_;
    MonitorElement* hist_Photon_Endcap_deltaPhi_;

    double hist_min_deltaPhi_;
    double hist_max_deltaPhi_;
    int    hist_bins_deltaPhi_;

    MonitorElement* hist_Photon_Barrel_deltaR_;
    MonitorElement* hist_Photon_Endcap_deltaR_;

    double hist_min_deltaR_;
    double hist_max_deltaR_;
    int    hist_bins_deltaR_;

    MonitorElement* hist_Photon_All_recoHMass_;
    MonitorElement* hist_Photon_BarrelOnly_recoHMass_;
    MonitorElement* hist_Photon_EndcapOnly_recoHMass_;
    MonitorElement* hist_Photon_Mixed_recoHMass_;

    double hist_min_recoHMass_;
    double hist_max_recoHMass_;
    int    hist_bins_recoHMass_;

    void findRecoHMass(reco::Photon pOne, reco::Photon pTwo);
};
#endif

