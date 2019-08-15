#ifndef WVALIDATION_H
#define WVALIDATION_H

/*class WValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class WValidation : public DQMEDAnalyzer {
public:
  explicit WValidation(const edm::ParameterSet &);
  ~WValidation() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) override;

private:
  WeightManager wmanager_;
  edm::InputTag hepmcCollection_;

  /// PDT table
  edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable;

  MonitorElement *nEvt;
  MonitorElement *Wmass, *WmassPeak, /* *WmT, *WmTPeak, */ *Wpt, *WptLog, *Wrap, *Wdaughters;
  MonitorElement *lepmet_mT, *lepmet_mTPeak, *lepmet_pt, *lepmet_ptLog, *lepmet_rap;
  MonitorElement *leppt, *met, *lepeta;
  MonitorElement *gamma_energy, *cos_theta_gamma_lepton;

  /// decay flavor
  int _flavor;
  /// decay flavor name
  std::string _name;

  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
};

#endif
