#ifndef BASICHEPMCVALIDATION_H
#define BASICHEPMCVALIDATION_H

/*class BasicHepMCValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2011/12/29 10:53:10 $
 *  $Revision: 1.6 $
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

class BasicHepMCValidation : public edm::EDAnalyzer
{
    public:
	explicit BasicHepMCValidation(const edm::ParameterSet&);
	virtual ~BasicHepMCValidation();
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

    MonitorElement* nEvt;
  
	///multiplicity ME's
	MonitorElement *uNumber, *dNumber, *sNumber, *cNumber, *bNumber, *tNumber;
	MonitorElement *ubarNumber, *dbarNumber, *sbarNumber, *cbarNumber, *bbarNumber, *tbarNumber;
	//
	MonitorElement *eminusNumber, *nueNumber, *muminusNumber, *numuNumber, *tauminusNumber, *nutauNumber;
	MonitorElement *eplusNumber, *nuebarNumber, *muplusNumber, *numubarNumber, *tauplusNumber, *nutaubarNumber;
	//
	MonitorElement *gluNumber, *WplusNumber,*WminusNumber, *ZNumber, *gammaNumber;
	//
	MonitorElement *piplusNumber, *piminusNumber, *pizeroNumber, *KplusNumber, *KminusNumber, *KlzeroNumber, *KszeroNumber;
	MonitorElement *pNumber, *pbarNumber, *nNumber, *nbarNumber, *l0Number, *l0barNumber;
	//
	MonitorElement *DplusNumber, *DminusNumber, *DzeroNumber, *BplusNumber, *BminusNumber, *BzeroNumber, *BszeroNumber;
	//
	MonitorElement *otherPtclNumber;
	
	///Momentum ME's
	MonitorElement *uMomentum, *dMomentum, *sMomentum, *cMomentum, *bMomentum, *tMomentum;
	MonitorElement *ubarMomentum, *dbarMomentum, *sbarMomentum, *cbarMomentum, *bbarMomentum, *tbarMomentum;
	//
	MonitorElement *eminusMomentum, *nueMomentum, *muminusMomentum, *numuMomentum, *tauminusMomentum, *nutauMomentum;
	MonitorElement *eplusMomentum, *nuebarMomentum, *muplusMomentum, *numubarMomentum, *tauplusMomentum, *nutaubarMomentum;
	//
	MonitorElement *gluMomentum, *WplusMomentum,*WminusMomentum, *ZMomentum, *gammaMomentum;
	//
	MonitorElement *piplusMomentum, *piminusMomentum, *pizeroMomentum, *KplusMomentum, *KminusMomentum, *KlzeroMomentum,  *KszeroMomentum;
	//
	MonitorElement *pMomentum, *pbarMomentum, *nMomentum, *nbarMomentum, *l0Momentum, *l0barMomentum;
	//
	MonitorElement *DplusMomentum, *DminusMomentum, *DzeroMomentum,  *BplusMomentum, *BminusMomentum, *BzeroMomentum, *BszeroMomentum;
	//
	MonitorElement *otherPtclMomentum;

	///other ME's
	MonitorElement *genPtclNumber; 
	MonitorElement *genVrtxNumber;
	MonitorElement *unknownPDTNumber;
	MonitorElement *outVrtxPtclNumber;
    MonitorElement *genPtclStatus;
    //
	MonitorElement *stablePtclNumber;
	MonitorElement *stableChaNumber;
	MonitorElement *stablePtclPhi;
	MonitorElement *stablePtclEta;
	MonitorElement *stablePtclCharge;
	MonitorElement *stablePtclp;
	MonitorElement *stablePtclpT;
	MonitorElement *partonNumber;
	MonitorElement *partonpT;
	MonitorElement *outVrtxStablePtclNumber;
	//
	MonitorElement *vrtxZ;
	MonitorElement *vrtxRadius;
	//
	MonitorElement *Bjorken_x;

    MonitorElement *status1ShortLived;

    MonitorElement *DeltaEcms;
    MonitorElement *DeltaPx;
    MonitorElement *DeltaPy;
    MonitorElement *DeltaPz;

};

#endif
