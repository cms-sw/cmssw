/* L1ExtraMaker Creates L1 Extra Objects from Clusters and jets

   S. Dutta  SINP-India*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1EGCrystalCluster.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"


class L1CaloGeometry;

namespace l1extra {
  class EtComparator {
  public:
    bool operator()(const l1extra::L1EmParticle a, const l1extra::L1EmParticle b) const {
      double et_a = 0.0;
      double et_b = 0.0;    
      if (cosh(a.eta()) > 0.0) et_a = a.energy()/cosh(a.eta());
      if (cosh(b.eta()) > 0.0) et_b = b.energy()/cosh(b.eta());
      
      return et_a > et_b;
    }
  };
  class ClusterETComparator {
  public:
    bool operator()(const l1slhc::L1EGCrystalCluster a, const l1slhc::L1EGCrystalCluster b) const {
      return a.pt() > b.pt();
    }
  };
}
class L1ExtraCrystalPosition:public edm::EDProducer {

public:
  explicit L1ExtraCrystalPosition( const edm::ParameterSet & );
  ~L1ExtraCrystalPosition(  );
  
private:
  
  virtual void produce( edm::Event &, const edm::EventSetup & );
  virtual void endJob(  );
  unsigned int getMatchedClusterIndex(l1slhc::L1EGCrystalClusterCollection& egxtals, float& eta, float& phi, float& dr_min);
  edm::InputTag mEgamma;
  edm::InputTag mEgCluster;
};

L1ExtraCrystalPosition::L1ExtraCrystalPosition( const edm::ParameterSet & iConfig ):
  mEgamma( iConfig.getParameter < edm::InputTag > ( "eGammaSrc" ) ),
  mEgCluster( iConfig.getParameter < edm::InputTag > ( "eClusterSrc" ) )

{
	// Register Product
	produces < l1extra::L1EmParticleCollection > ( "EGammaCrystal" );
}


L1ExtraCrystalPosition::~L1ExtraCrystalPosition(  )
{
}


void L1ExtraCrystalPosition::produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) {
  edm::Handle<l1slhc::L1EGCrystalClusterCollection> egCrystalClusters;
  iEvent.getByLabel(mEgCluster,egCrystalClusters);
  l1slhc::L1EGCrystalClusterCollection clusColl = (*egCrystalClusters.product());
  std::sort(clusColl.begin(), clusColl.end(), l1extra::ClusterETComparator());

  edm::Handle < l1extra::L1EmParticleCollection > eg;
  std::auto_ptr < l1extra::L1EmParticleCollection > l1EGammaCrystal( new l1extra::L1EmParticleCollection );

  if ( iEvent.getByLabel( mEgamma, eg ) ) {
    l1extra::L1EmParticleCollection egColl = (*eg.product());
    std::sort(egColl.begin(), egColl.end(), l1extra::EtComparator());
    std::vector<unsigned int> matched_indices;
    for ( l1extra::L1EmParticleCollection::const_iterator lIt =egColl.begin(); lIt != egColl.end(); ++lIt ) {
      l1extra::L1EmParticle em( *lIt );
      float eta = em.eta();
      float phi = em.phi();
      //      float et = 0;
      //      if (cosh(eta) > 0.0) et = em.energy()/cosh(eta);
      //      std::cout << " Et, Energy, Eta, Phi (before correction) " << et << " " << em.energy() << " " << eta << " " << phi << std::endl;
      float deltaR;
      unsigned int indx = getMatchedClusterIndex(clusColl, eta, phi, deltaR);
      if (indx != 9999 && deltaR < 0.3) {
	l1extra::L1EmParticle* em_new = em.clone();
        float eta_new = clusColl[indx].eta();
        float phi_new = clusColl[indx].phi();
	//	float pt_new  = em.pt();
	//	if (cosh(eta_new) > 0.0) pt_new = em.p()/cosh(eta_new);
	//	reco::Candidate::PolarLorentzVector lv_em_corr(pt_new, eta_new, phi_new, em.mass());
	//        em_new->setP4(lv_em_corr);
	//	reco::Candidate::PolarLorentzVector & p4 = const_cast<reco::Candidate::PolarLorentzVector &> (em_new->polarP4());
	//        p4.SetPhi(phi_new);
	//        p4.SetEta(eta_new);

	reco::Candidate::PolarLorentzVector p4(em.pt(), eta_new, phi_new, em.mass());
        em_new->setP4(p4);       
	l1EGammaCrystal->push_back( *em_new );   
	//	float et_new = 0;
	//	if (cosh(em_new->eta()) > 0.0) et_new = em_new->energy()/cosh(em_new->eta());
	//	std::cout << " Et, Energy, Eta, Phi (after correction) " << et_new << " " << em_new->energy() << " " << em_new->eta() << " " << em_new->phi() << " deltaR " << deltaR << std::endl;         
	//	std::cout << " px, py, pz " <<  em_new->px() << " " << em_new->py() << " " << em_new->pz() << std::endl;         
      }
    }
    iEvent.put(l1EGammaCrystal, "EGammaCrystal" );
  }
}

// ------------ method called once each job just after ending the event loop ------------
void L1ExtraCrystalPosition::endJob(  )
{
}
unsigned int L1ExtraCrystalPosition::getMatchedClusterIndex(l1slhc::L1EGCrystalClusterCollection& egxtals, float& eta, float& phi, float& dr_min) {
  dr_min = 999.9;
  size_t index_min = 9999;
  for (size_t i = 0; i < egxtals.size(); i++) {
    float dr = deltaR(egxtals[i].eta(), egxtals[i].phi(), eta, phi);
    if (dr < dr_min) {
      index_min = i;
      dr_min = dr;
    }
  }
  return index_min; 
}

DEFINE_FWK_MODULE(L1ExtraCrystalPosition);
