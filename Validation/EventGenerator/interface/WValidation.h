#ifndef WVALIDATION_H
#define WVALIDATION_H

/*class WValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2011/12/29 10:53:10 $
 *  $Revision: 1.3 $
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
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"


class WValidation : public edm::EDAnalyzer
{
  public:
	explicit WValidation(const edm::ParameterSet&);
	virtual ~WValidation();
	virtual void beginJob();
	virtual void endJob();  
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void beginRun(const edm::Run&, const edm::EventSetup&);
	virtual void endRun(const edm::Run&, const edm::EventSetup&);

  private:

  WeightManager _wmanager;

  edm::InputTag hepmcCollection_;

  /// PDT table
  edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
  
  ///ME's "container"
  DQMStore *dbe;

  MonitorElement *nEvt;  
  MonitorElement *Wmass, *WmassPeak, /* *WmT, *WmTPeak, */  *Wpt, *WptLog, *Wrap, *Wdaughters;
  MonitorElement *lepmet_mT, *lepmet_mTPeak, *lepmet_pt, *lepmet_ptLog, *lepmet_rap;
  MonitorElement *leppt, *met, *lepeta;
  MonitorElement *gamma_energy, *cos_theta_gamma_lepton; 

  /// decay flavor
  int _flavor;
  /// decay flavor name
  std::string _name;

};

#endif
