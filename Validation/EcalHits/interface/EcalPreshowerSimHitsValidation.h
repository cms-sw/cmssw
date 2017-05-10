#ifndef EcalPreshowerSimHitsValidation_H
#define EcalPreshowerSimHitsValidation_H

/*
 * \file EcalPreshowerSimHitsValidation.h
 *
 * \author C.Rovelli
 *
*/

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include <map>

class MonitorElement;

class EcalPreshowerSimHitsValidation : public DQMEDAnalyzer {

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

 public:

  /// Constructor
  EcalPreshowerSimHitsValidation(edm::ParameterSet const&);

  /// Destructor
  ~EcalPreshowerSimHitsValidation();

 protected:

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

 private:

  edm::EDGetTokenT<edm::HepMCProduct> HepMCToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EEHitsToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> ESHitsToken;
 
  MonitorElement* menESHits1zp_;
  MonitorElement* menESHits2zp_;

  MonitorElement* menESHits1zm_;
  MonitorElement* menESHits2zm_;

  MonitorElement* meEShitLog10Energy_;
  MonitorElement* meEShitLog10EnergyNorm_;

  MonitorElement* meESEnergyHits1zp_;
  MonitorElement* meESEnergyHits2zp_;

  MonitorElement* meESEnergyHits1zm_;
  MonitorElement* meESEnergyHits2zm_;

  MonitorElement* meE1alphaE2zp_;
  MonitorElement* meE1alphaE2zm_;

  MonitorElement* meEEoverESzp_;
  MonitorElement* meEEoverESzm_;

  MonitorElement* me2eszpOver1eszp_; 
  MonitorElement* me2eszmOver1eszm_; 

};

#endif
