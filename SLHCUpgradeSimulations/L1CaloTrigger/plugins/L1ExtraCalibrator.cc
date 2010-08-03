/* L1ExtraMaker
Creates L1 Extra Objects from Clusters and jets

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/


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

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

class L1CaloGeometry;

class L1ExtraCalibrator : public edm::EDProducer {
   public:
      explicit L1ExtraCalibrator(const edm::ParameterSet&);
      ~L1ExtraCalibrator();

   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag eGamma_;
      edm::InputTag isoEGamma_;
      edm::InputTag taus_;
      edm::InputTag isoTaus_;
      edm::InputTag jets_;

  std::vector<double> eGammaCoeff_;
  std::vector<double> tauCoeff_;

  math::PtEtaPhiMLorentzVector calibratedP4(const math::PtEtaPhiMLorentzVector&,const  std::vector<double>&);


};





L1ExtraCalibrator::L1ExtraCalibrator(const edm::ParameterSet& iConfig):
  eGamma_(iConfig.getParameter<edm::InputTag>("eGamma")),
  isoEGamma_(iConfig.getParameter<edm::InputTag>("isoEGamma")),
  taus_(iConfig.getParameter<edm::InputTag>("taus")),
  isoTaus_(iConfig.getParameter<edm::InputTag>("isoTaus")),
  jets_(iConfig.getParameter<edm::InputTag>("jets")),
  eGammaCoeff_(iConfig.getParameter<std::vector<double> >("eGammaCoefficients")),
  tauCoeff_(iConfig.getParameter<std::vector<double> >("tauCoefficients"))
{
  //Register Product
  produces<l1extra::L1EmParticleCollection>("EGamma");
  produces<l1extra::L1EmParticleCollection>("IsoEGamma");
  produces<l1extra::L1JetParticleCollection>("Taus");
  produces<l1extra::L1JetParticleCollection>("IsoTaus");
  produces<l1extra::L1JetParticleCollection>("Jets");

}


L1ExtraCalibrator::~L1ExtraCalibrator()
{}


void
L1ExtraCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace l1slhc;
   using namespace l1extra;




   std::auto_ptr<L1EmParticleCollection>  l1EGamma(new L1EmParticleCollection);
   std::auto_ptr<L1EmParticleCollection>  l1IsoEGamma(new L1EmParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1Tau(new L1JetParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1IsoTau(new L1JetParticleCollection);
   std::auto_ptr<L1JetParticleCollection>  l1Jet(new L1JetParticleCollection);


   edm::Handle<L1EmParticleCollection> eg;
   if(iEvent.getByLabel(eGamma_,eg)) 
     for(unsigned int i=0;i<eg->size();++i) {
       L1EmParticle p = eg->at(i);
       p.setP4(calibratedP4(p.polarP4(),eGammaCoeff_));
       l1EGamma->push_back(p);
     }

   edm::Handle<L1EmParticleCollection> ieg;
   if(iEvent.getByLabel(isoEGamma_,ieg)) 
     for(unsigned int i=0;i<ieg->size();++i) {
       L1EmParticle p = ieg->at(i);
       p.setP4(calibratedP4(p.polarP4(),eGammaCoeff_));
       l1IsoEGamma->push_back(p);
     }

   edm::Handle<L1JetParticleCollection> tau;
   if(iEvent.getByLabel(taus_,tau)) 
     for(unsigned int i=0;i<tau->size();++i) {
       L1JetParticle p = tau->at(i);
       p.setP4(calibratedP4(p.polarP4(),tauCoeff_));
       l1Tau->push_back(p);
     }

   edm::Handle<L1JetParticleCollection> itau;
   if(iEvent.getByLabel(isoTaus_,itau)) 
     for(unsigned int i=0;i<itau->size();++i) {
       L1JetParticle p = itau->at(i);
       p.setP4(calibratedP4(p.polarP4(),tauCoeff_));
       l1IsoTau->push_back(p);
     }

   edm::Handle<L1JetParticleCollection> jets;
   if(iEvent.getByLabel(jets_,jets)) 
     for(unsigned int i=0;i<jets->size();++i) {
       L1JetParticle p = jets->at(i);
       l1Jet->push_back(p);
     }


   iEvent.put(l1EGamma,"EGamma");
   iEvent.put(l1IsoEGamma,"IsoEGamma");
   iEvent.put(l1Tau,"Taus");
   iEvent.put(l1IsoTau,"IsoTaus");
   iEvent.put(l1Jet,"Jets");


}
// ------------ method called once each job just after ending the event loop  ------------
void
L1ExtraCalibrator::endJob() {
}


math::PtEtaPhiMLorentzVector 
L1ExtraCalibrator::calibratedP4(const math::PtEtaPhiMLorentzVector& p4,const  std::vector<double>& coeffs)
{
  
  double factor = coeffs.at(0)+coeffs.at(1)*fabs(p4.eta()) + coeffs.at(2)*p4.eta()*p4.eta();
  math::PtEtaPhiMLorentzVector calibrated(factor*p4.pt(),p4.eta(),p4.phi(),0.0); 
  return calibrated;
}


//#define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1ExtraCalibrator>,"L1ExtraCalibrator"); DEFINE_FWK_PSET_DESC_FILLER(L1ExtraCalibrator);
//DEFINE_ANOTHER_FWK_MODULE(L1ExtraCalibrator);

