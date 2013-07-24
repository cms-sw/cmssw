#ifndef DRELLYANVALIDATION_H
#define DRELLYANVALIDATION_H

/*class DrellYanValidation
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

class DrellYanValidation : public edm::EDAnalyzer
{
  public:
	explicit DrellYanValidation(const edm::ParameterSet&);
	virtual ~DrellYanValidation();
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
  MonitorElement *Zmass, *ZmassPeak, *Zpt, *ZptLog, *Zrap, *Zdaughters;
  MonitorElement *dilep_mass, *dilep_massPeak, *dilep_pt, *dilep_ptLog, *dilep_rap;
  MonitorElement *leadpt, *secpt, *leadeta, *seceta;
  MonitorElement *gamma_energy, *cos_theta_gamma_lepton; 

  /// decay flavor
  int _flavor;
  /// decay flavor name
  std::string _name;

};

#endif
