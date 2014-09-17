#ifndef EcalSimHitsValidation_H
#define EcalSimHitsValidation_H

/*
 * \file EcalSimHitsValidation.h
 *
 * \author C.Rovelli
 *
*/

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <map>

class MonitorElement;

class EcalSimHitsValidation : public DQMEDAnalyzer {

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

 public:

  /// Constructor
  EcalSimHitsValidation(edm::ParameterSet const&);

  /// Destructor
  ~EcalSimHitsValidation();

 protected:

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&);

 private:

  edm::EDGetTokenT<edm::HepMCProduct> HepMCToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EBHitsCollectionToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EEHitsCollectionToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> ESHitsCollectionToken;

  MonitorElement* meGunEnergy_;
  MonitorElement* meGunEta_;
  MonitorElement* meGunPhi_;

  MonitorElement* meEBEnergyFraction_;
  MonitorElement* meEEEnergyFraction_;
  MonitorElement* meESEnergyFraction_;
};

#endif
