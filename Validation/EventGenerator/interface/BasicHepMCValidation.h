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
#include "Validation/EventGenerator/interface/DQMHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/WeightManager.h"
#include "TVector3.h"

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
    ParticleMonitor(std::string name_,int pdgid_, DQMStore::IBooker &i,bool nlog_=false):name(name_),pdgid(pdgid_),count(0),nlog(nlog_){
	DQMHelper dqm(&i);
	// Number of analyzed events
	if(!nlog){
	numberPerEvent= dqm.book1dHisto(name+"Number", "Number of  "+name+"'s per event", 
					20, 0, 20,"No. of "+name,"Number of Events");
	}
	else{
	  numberPerEvent= dqm.book1dHisto(name+"Number", "Number of  "+name+"'s per event",
					  20, 0, 20,"log_{10}(No. of "+name+")","Number of Events");
	}
	p_init  = dqm.book1dHisto(name+"Momentum", "log_{10}(P) of the "+name+"s", 
				  60, -2, 4,"log_{10}(P) (log_{10}(GeV))","Number of "+name );
	
	eta_init  = dqm.book1dHisto(name+"Eta", "#eta of the "+name+"s", 
				   100, -5., 5.,"#eta","Number of "+name);

	lifetime_init  = dqm.book1dHisto(name+"LifeTime", "#phi of the "+name+"s", 
					 100, -15, -5,"Log_{10}(life-time^{final}) (log_{10}(s))","Number of "+name);
	
	p_final = dqm.book1dHisto(name+"MomentumFinal", "log_{10}(P^{final}) of "+name+"s at end of decay chain", 
				  60, -2, 4,"log_{10}(P^{final}) (log_{10}(GeV))","Number of "+name);
	
	lifetime_final=dqm.book1dHisto(name+"LifeTimeFinal", "Log_{10}(life-time^{final}) of "+name+"s at end of decay chain",
					100,-15,-5,"Log_{10}(life-time^{final}) (log_{10}(s))","Number of "+name);
      }

      ~ParticleMonitor(){};
      
      bool Fill(const HepMC::GenParticle* p, double weight){
	if(p->pdg_id()==pdgid){
	  if(isFirst(p)){
	    p_init->Fill(log10(p->momentum().rho()),weight);
	    eta_init->Fill(p->momentum().eta(),weight);
            const HepMC::GenParticle* pf=GetFinal(p); // inlcude mixing
            p_final->Fill(log10(pf->momentum().rho()),weight);
	    // compute lifetime...
	    if(p->production_vertex() && p->end_vertex()){
	      TVector3 PV(p->production_vertex()->point3d().x(),p->production_vertex()->point3d().y(),p->production_vertex()->point3d().z()); 
	      TVector3 SV(p->end_vertex()->point3d().x(),p->end_vertex()->point3d().y(),p->end_vertex()->point3d().z()); 
	      TVector3 DL=SV-PV; 
	      double c(2.99792458E8),Ltau(DL.Mag()/100)/*cm->m*/,beta(p->momentum().rho()/p->momentum().m()); 
	      double lt=Ltau/(c*beta);
	      if(lt>1E-16)lifetime_init->Fill(log10(lt),weight);
	      if(pf->end_vertex()){
		TVector3 SVf(pf->end_vertex()->point3d().x(),pf->end_vertex()->point3d().y(),pf->end_vertex()->point3d().z());
		DL=SVf-PV;
		Ltau=DL.Mag()/100;
		lt=Ltau/(c*beta);
		if(lt>1E-16)lifetime_final->Fill(log10(lt),weight);
	      }
	    }
	    count++;
	  }
	  return true;
	}
	return false;
      }
      
      void FillCount(double weight){
	if(nlog) numberPerEvent->Fill(log10(count),weight);
	else numberPerEvent->Fill(count,weight);
	count=0;
      }
      
      int PDGID(){return pdgid;}
      
    private:
      bool isFirst(const HepMC::GenParticle* p){
	if(p->production_vertex()){
	  for(HepMC::GenVertex::particles_in_const_iterator m=p->production_vertex()->particles_in_const_begin(); m!=p->production_vertex()->particles_in_const_end();m++){
	    if(abs((*m)->pdg_id())==abs(p->pdg_id())) return false;
	  }
	}
	return true;
      }
      
      const HepMC::GenParticle* GetFinal(const HepMC::GenParticle* p){ // includes mixing (assuming mixing is not occurring more than 5 times back and forth)
        HepMC::GenParticle* aPart = new HepMC::GenParticle(*p);
	for (unsigned int iMix = 0; iMix < 10; iMix++) {
	  bool foundSimilar = false;
	  if(aPart->end_vertex()){ 
	    if(aPart->end_vertex()->particles_out_size()!=0){ 
	      for(HepMC::GenVertex::particles_out_const_iterator d=aPart->end_vertex()->particles_out_const_begin(); d!=aPart->end_vertex()->particles_out_const_end();d++){ 
		if(abs((*d)->pdg_id())==abs(aPart->pdg_id())){ 
		  aPart = *d;         
		  foundSimilar = true;
		  break;
		} 
	      } 
	    }
	    if (!foundSimilar) break;
	  } 
	} 
	return aPart;
      }
      
      std::string name;
      int pdgid;
      unsigned int count;
      bool nlog;
      MonitorElement *p_init, *p_final, *eta_init, *lifetime_init, *lifetime_final, *numberPerEvent;
    };
    
    
    MonitorElement* nEvt;
    std::vector<ParticleMonitor> particles;
    
    ///other ME's
    MonitorElement *otherPtclNumber;
    MonitorElement *otherPtclMomentum;
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
