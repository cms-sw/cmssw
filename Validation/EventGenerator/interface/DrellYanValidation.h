#ifndef DRELLYANVALIDATION_H
#define DRELLYANVALIDATION_H

/*class DrellYanValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *
 */

// framework & common header files
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

class DrellYanValidation : public DQMEDAnalyzer {
public:
  explicit DrellYanValidation(const edm::ParameterSet &);
  ~DrellYanValidation() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) override;
  void analyze(edm::Event const &, edm::EventSetup const &) override;

private:
  WeightManager wmanager_;
  edm::InputTag hepmcCollection_;

  /// PDT table
  edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable;

  MonitorElement *nEvt;
  MonitorElement *Zmass, *ZmassPeak, *Zpt, *ZptLog, *Zrap, *Zdaughters;
  MonitorElement *dilep_mass, *dilep_massPeak, *dilep_pt, *dilep_ptLog, *dilep_rap;
  MonitorElement *leadpt, *secpt, *leadeta, *seceta;
  MonitorElement *gamma_energy, *cos_theta_gamma_lepton;

  /// decay flavor
  int _flavor;
  /// decay flavor name
  std::string _name;

  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
};

#endif
