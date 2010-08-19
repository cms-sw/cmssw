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
  
  std::vector<double> eGammaCoeffB_;
  std::vector<double> tauCoeffB_;
  std::vector<double> eGammaCoeffE_;
  std::vector<double> tauCoeffE_;
  std::vector<double> eGammaBinCorr_;
  std::vector<double> tauBinCorr_;


  math::PtEtaPhiMLorentzVector calibratedP4(const math::PtEtaPhiMLorentzVector&,const  std::vector<double>&, const std::vector<double>&);


};





L1ExtraCalibrator::L1ExtraCalibrator(const edm::ParameterSet& iConfig):
  eGamma_(iConfig.getParameter<edm::InputTag>("eGamma")),
  isoEGamma_(iConfig.getParameter<edm::InputTag>("isoEGamma")),
  taus_(iConfig.getParameter<edm::InputTag>("taus")),
  isoTaus_(iConfig.getParameter<edm::InputTag>("isoTaus")),
  jets_(iConfig.getParameter<edm::InputTag>("jets")),
  eGammaCoeffB_(iConfig.getParameter<std::vector<double> >("eGammaCoefficientsB")),
  tauCoeffB_(iConfig.getParameter<std::vector<double> >("tauCoefficientsB")),
  eGammaCoeffE_(iConfig.getParameter<std::vector<double> >("eGammaCoefficientsE")),
  tauCoeffE_(iConfig.getParameter<std::vector<double> >("tauCoefficientsE")),
  eGammaBinCorr_(iConfig.getParameter<std::vector<double> >("eGammaBinCorr")),
  tauBinCorr_(iConfig.getParameter<std::vector<double> >("tauBinCorr"))
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
   if(iEvent.getByLabel(eGamma_,eg)) {
     for(unsigned int i=0;i<eg->size();++i) {
       L1EmParticle p = eg->at(i);
       //Pass E or B coefficients depending on p.eta().
       if (fabs(p.eta())<1.6){
	 p.setP4(calibratedP4(p.polarP4(),eGammaCoeffB_,eGammaBinCorr_));
       } else if (fabs(p.eta())<2.6){
	 p.setP4(calibratedP4(p.polarP4(),eGammaCoeffE_,eGammaBinCorr_));
       }
       l1EGamma->push_back(p);
     }
   }

   edm::Handle<L1EmParticleCollection> ieg;
   if(iEvent.getByLabel(isoEGamma_,ieg)) {
     for(unsigned int i=0;i<ieg->size();++i) {
       L1EmParticle p = ieg->at(i);
       if (fabs(p.eta())<1.6){
	 p.setP4(calibratedP4(p.polarP4(),eGammaCoeffB_,eGammaBinCorr_));
       } else if (fabs(p.eta())<2.6){
	 p.setP4(calibratedP4(p.polarP4(),eGammaCoeffE_,eGammaBinCorr_));
       }
       l1IsoEGamma->push_back(p);
     }
   }
   
   edm::Handle<L1JetParticleCollection> tau;
   if(iEvent.getByLabel(taus_,tau)) {
     for(unsigned int i=0;i<tau->size();++i) {
       L1JetParticle p = tau->at(i);
       if (fabs(p.eta())<1.6){
	 p.setP4(calibratedP4(p.polarP4(),tauCoeffB_,tauBinCorr_));
       } else if (fabs(p.eta())<2.6){
	 p.setP4(calibratedP4(p.polarP4(),tauCoeffE_,tauBinCorr_));
       }
       l1Tau->push_back(p);
     }
   }
   
   edm::Handle<L1JetParticleCollection> itau;
   if(iEvent.getByLabel(isoTaus_,itau)) {
     for(unsigned int i=0;i<itau->size();++i) {
       L1JetParticle p = itau->at(i);
       if (fabs(p.eta())<1.6){
	 p.setP4(calibratedP4(p.polarP4(),tauCoeffB_,tauBinCorr_));
       } else if (fabs(p.eta())<2.6){
	 p.setP4(calibratedP4(p.polarP4(),tauCoeffE_,tauBinCorr_));
       }
       l1IsoTau->push_back(p);
     }
   }

   edm::Handle<L1JetParticleCollection> jets;
   if(iEvent.getByLabel(jets_,jets)) {
     for(unsigned int i=0;i<jets->size();++i) {
       L1JetParticle p = jets->at(i);
       l1Jet->push_back(p);
     }
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
L1ExtraCalibrator::calibratedP4(const math::PtEtaPhiMLorentzVector& p4,const  std::vector<double>& coeffs, const std::vector<double>& binCorrs)
{
  
  //    double factor = coeffs.at(0)+coeffs.at(1)*fabs(p4.eta()) + coeffs.at(2)*p4.eta()*p4.eta();
  double factor = 0;
  double bfactor = 0;
    if ( fabs(p4.eta())<1.6 ){
      factor = coeffs.at(0) + coeffs.at(1)*fabs(p4.eta())+ coeffs.at(2)*fabs(p4.eta())*fabs(p4.eta());
      //      printf("factorB is %f \n",factor);
    } else if ( fabs(p4.eta())<2.5 ){
      factor = coeffs.at(0) + coeffs.at(1)*(fabs(p4.eta())-1.6)+ coeffs.at(2)*(fabs(p4.eta())-1.6)*(fabs(p4.eta())-1.6);
      //printf("factorE is %f \n",factor);
    }

  // apply bin-by-bin corrections
  if (fabs(p4.eta())>=0 && fabs(p4.eta())<0.2){
    bfactor=binCorrs.at(0);
    printf("bin 0: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=0.2 && fabs(p4.eta())<0.4){
    bfactor=binCorrs.at(1);
    printf("bin 1: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=0.4 && fabs(p4.eta())<0.6){
    bfactor=binCorrs.at(2);
    printf("bin 2: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=0.6 && fabs(p4.eta())<0.8){
    bfactor=binCorrs.at(3);
    printf("bin 3: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=0.8 && fabs(p4.eta())<1.0){
    bfactor=binCorrs.at(4);
    printf("bin 4: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=1.0 && fabs(p4.eta())<1.2){
    bfactor=binCorrs.at(5);
    printf("bin 5: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=1.2 && fabs(p4.eta())<1.4){
    bfactor=binCorrs.at(6);
    printf("bin 6: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=1.4 && fabs(p4.eta())<1.6){
    bfactor=binCorrs.at(7);
    printf("bin 7: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=1.6 && fabs(p4.eta())<1.8){
    bfactor=binCorrs.at(8);
    printf("bin 8: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=1.8 && fabs(p4.eta())<2.0){
    bfactor=binCorrs.at(9);
    printf("bin 9: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=2.0 && fabs(p4.eta())<2.2){
    bfactor=binCorrs.at(10);
    printf("bin 10: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=2.2 && fabs(p4.eta())<2.4){
    bfactor=binCorrs.at(11);
    printf("bin 11: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else if (fabs(p4.eta())>=2.4 && fabs(p4.eta())<2.6){
    bfactor=binCorrs.at(12);
    printf("bin 12: corr. of %f\t abs(eta) of %f \n",bfactor,fabs(p4.eta()));
  } else {
    bfactor=1.0;
    printf("no bin found! Corr. set to 1.0");
  }
  math::PtEtaPhiMLorentzVector calibrated(factor*bfactor*p4.pt(),p4.eta(),p4.phi(),0.0);

  return calibrated;
}


//#define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<L1ExtraCalibrator>,"L1ExtraCalibrator"); DEFINE_FWK_PSET_DESC_FILLER(L1ExtraCalibrator);
//DEFINE_ANOTHER_FWK_MODULE(L1ExtraCalibrator);

