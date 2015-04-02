#ifndef BASICHEPMCVALIDATION_H
#define BASICHEPMCVALIDATION_H

/*class BasicHepMCValidation
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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"

class BasicHepMCValidation : public DQMEDAnalyzer{
    public:
	explicit BasicHepMCValidation(const edm::ParameterSet&);
	virtual ~BasicHepMCValidation();

        virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
        virtual void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;
        virtual void analyze(edm::Event const&, edm::EventSetup const&) override;

    private:
	WeightManager wmanager_;
    edm::InputTag hepmcCollection_;

    /// PDT table
    edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
    
    
    class ParticleMonitor{
    public:
    ParticleMonitor(TString _name,int pdgid_):name(_name),pdgid(pdgid_),count(0){};
      ~ParticleMonitor(){};
      
      void Configure(DQMStore::IBooker &i){
	TString pname=p.getParameter<std::string>("pname");
	double mass_min=p.getParameter<double>("massmin");
	double mass_max=p.getParameter<double>("massmax");
	DQMHelper dqm(&i); i.setCurrentFolder("Generator/BPhysics");
	// Number of analyzed events
	pt   = dqm.book1dHisto(name+"PT", "P_{t} of the "+pname+"s", 100, 0., 100,"P_{t} (GeV)","Number of Events");
	eta  = dqm.book1dHisto(name+"ETA", "#eta of the "+pname+"s", 100, -5., 5.,"#eta","Number of Events");
	phi  = dqm.book1dHisto(name+"PHI", "#phi of the "+pname+"s", 100, 0, 2*TMath::Pi(),"#phi","Number of Events");
	mass = dqm.book1dHisto(name+"MASS", "Mass of the "+pname+"s", 100, mass_min, mass_max,"Mass (GeV)","Number of Events");
      }
      
      void Fill(const reco::GenParticle* p, double weight){
	if(abs(p->pdgId())==abs(pdgid)){
	  if(isFirst(p)){
	    pt_init->Fill(p->pt(),weight);
	    eta_init->Fill(p->eta(),weight);
	    lifetime->Fill(lt,weight);
	    pf=GetFinal(p);
	    pt_final->Fill(pf->pt(),weight);
	    count++;
	  }
	}
      }
      
      void FillCount(double weight){
	numberPerEvent->Fill(count,weight);
	count=0;
      }
	  
      int PDGID(){return pdgid;}
      
    private:
      bool isFirst(const reco::GenParticle* p){
	
      }
      
      reco::GenParticle* GetFinal(const reco::GenParticle* p){
	
      }
      
      TString name;
      int pdgid;
      unsigned int count;
      MonitorElement *pt_init, *pt_final, *eta_init, *lifetime, *numberPerEvent;
    };
    
	
    MonitorElement* nEvt;
    std::vector<ParticleMonitor> particle;
    
    
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
    
    MonitorElement *log10DeltaEcms;
    MonitorElement *DeltaEcms;
    MonitorElement *DeltaPx;
    MonitorElement *DeltaPy;
    MonitorElement *DeltaPz;
    
    edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;
    
};

#endif
