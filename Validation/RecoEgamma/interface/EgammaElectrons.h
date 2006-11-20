#ifndef Validation_RecoEgamma_EgammaElectrons_h
#define Validation_RecoEgamma_EgammaElectrons_h

/**\class EgammaElectrons

 Description: SVSuite Electron Validation

 Implementation:
     \\\author: Michael A. Balazs, Nov 2006
*/
//
// $Id: EgammaElectrons.h,v 1.3 2006/11/14 21:35:45 mabalazs Exp $
//
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

class EgammaElectrons : public edm::EDAnalyzer
{
	public:
      explicit EgammaElectrons( const edm::ParameterSet& );
      ~EgammaElectrons();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();

	private:
			std::string outputFile_;
			std::string CMSSW_Version_;

			bool verboseDBE_;
			DaqMonitorBEInterface* dbe_;

      edm::InputTag MCTruthCollection_;
      edm::InputTag ElectronCollection_;
 
      MonitorElement* hist_Electron_Size_;

      double hist_min_Size_;
      double hist_max_Size_;
      int    hist_bins_Size_;

      MonitorElement* hist_Electron_Barrel_ET_;
      MonitorElement* hist_Electron_Endcap_ET_;

      double hist_min_ET_;
      double hist_max_ET_;
      int    hist_bins_ET_;

      MonitorElement* hist_Electron_Barrel_Eta_;
      MonitorElement* hist_Electron_Endcap_Eta_;

      double hist_min_Eta_;
      double hist_max_Eta_;
      int    hist_bins_Eta_;

      MonitorElement* hist_Electron_Barrel_Phi_;
      MonitorElement* hist_Electron_Endcap_Phi_;

      double hist_min_Phi_;
      double hist_max_Phi_;
      int    hist_bins_Phi_;

      MonitorElement* hist_Electron_Barrel_EoverP_;
      MonitorElement* hist_Electron_Endcap_EoverP_;

      double hist_min_EoverP_;
      double hist_max_EoverP_;
      int    hist_bins_EoverP_;

      MonitorElement* hist_Electron_Barrel_deltaEtaSCtoTrack_;
      MonitorElement* hist_Electron_Endcap_deltaEtaSCtoTrack_;

      double hist_min_deltaEtaSCtoTrack_;
      double hist_max_deltaEtaSCtoTrack_;
      int    hist_bins_deltaEtaSCtoTrack_;


      MonitorElement* hist_Electron_Barrel_EToverTruth_;
      MonitorElement* hist_Electron_Endcap_EToverTruth_;

      double hist_min_EToverTruth_;
      double hist_max_EToverTruth_;
      int    hist_bins_EToverTruth_;

      MonitorElement* hist_Electron_Barrel_deltaEta_;
      MonitorElement* hist_Electron_Endcap_deltaEta_;

      double hist_min_deltaEta_;
      double hist_max_deltaEta_;
      int    hist_bins_deltaEta_;

      MonitorElement* hist_Electron_Barrel_deltaPhi_;
      MonitorElement* hist_Electron_Endcap_deltaPhi_;

      double hist_min_deltaPhi_;
      double hist_max_deltaPhi_;
      int    hist_bins_deltaPhi_;

      MonitorElement* hist_Electron_Barrel_deltaR_;
      MonitorElement* hist_Electron_Endcap_deltaR_;

      double hist_min_deltaR_;
      double hist_max_deltaR_;
      int    hist_bins_deltaR_;
};
#endif

