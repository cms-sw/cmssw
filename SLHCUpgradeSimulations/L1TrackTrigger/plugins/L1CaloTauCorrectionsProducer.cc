// -*- C++ -*-
//
// Package:    L1CaloTauCorrectionsProducer
// Class:      L1CaloTauCorrectionsProducer
// 
/**\class L1CaloTauCorrectionsProducer L1CaloTauCorrectionsProducer.cc SLHCUpgradeSimulations/L1CaloTauCorrectionsProducer/plugins/L1CaloTauCorrectionsProducer.cc

   Description: A very trivial producer which takes as input a collections for the stage-2 L1CaloTaus and produces another identical collection with calibration
   corrections applied to the 4-momenta of each L1CaloTau. 

   Implementation: To be used for L1 Menu (Rate) calculations of stage-2 L1CaloTaus. 
 
*/
//
// Original Author:  Alexandros Attikis
//         Created:  Tue, 10 Jun 2014 09:11:12 GMT
// $Id$
//
//


// system include files
#include <memory>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

//
// class declaration
//

class L1CaloTauCorrectionsProducer : public edm::EDProducer {
public:

typedef std::vector< l1extra::L1JetParticle > L1CaloTauCollectionType;

explicit L1CaloTauCorrectionsProducer(const edm::ParameterSet&);
~L1CaloTauCorrectionsProducer();
  
static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
virtual void beginJob() override;
virtual void produce(edm::Event&, const edm::EventSetup&) override;
virtual void endJob() override;      
double GetCorrectionFactor(double Et, double Eta);
int FindNearestIndex(const unsigned int arraySize, double myVector[], double myEtaValue);

// ----------member data ---------------------------
edm::InputTag L1TausInputTag;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloTauCorrectionsProducer::L1CaloTauCorrectionsProducer(const edm::ParameterSet& iConfig)
{

  
// register your products
L1TausInputTag  = iConfig.getParameter<edm::InputTag>("L1TausInputTag");

produces<L1CaloTauCollectionType>("CalibratedTaus").setBranchAlias( "CalibratedTaus");
  
}


L1CaloTauCorrectionsProducer::~L1CaloTauCorrectionsProducer()
{

}

// ------------ method called to produce the data  ------------
void
L1CaloTauCorrectionsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
using namespace edm;

std::auto_ptr<L1CaloTauCollectionType> result(new L1CaloTauCollectionType);

// Get the stage-2 L1CaloTaus from the L1-ExtraParticles
edm::Handle< L1CaloTauCollectionType > h_L1CaloTau;
iEvent.getByLabel( L1TausInputTag, h_L1CaloTau );

// for-loop: L1CaloTaus
for ( L1CaloTauCollectionType::const_iterator L1CaloTau = h_L1CaloTau->begin();  L1CaloTau != h_L1CaloTau->end(); L1CaloTau++){

double L1CaloTau_E   = L1CaloTau->p4().energy();
double L1CaloTau_Px  = L1CaloTau->p4().px();
double L1CaloTau_Py  = L1CaloTau->p4().py();
double L1CaloTau_Pz  = L1CaloTau->p4().pz();

//math::XYZTLorentzVector p4 = L1CaloTau->p4();
math::XYZTLorentzVector p4Corr(0, 0, 0, 0);

double corrFactor = L1CaloTauCorrectionsProducer::GetCorrectionFactor( L1CaloTau->p4().Et(), L1CaloTau->p4().eta() );
p4Corr.SetPxPyPzE( L1CaloTau_Px*corrFactor, L1CaloTau_Py*corrFactor, L1CaloTau_Pz*corrFactor, L1CaloTau_E*corrFactor);
l1extra::L1JetParticle L1CaloTauCorr(p4Corr);

// std::cout << "---> p4.energy() = " << p4.energy() << ", p4Corr.energy() = " << p4Corr.energy() << std::endl;
result -> push_back( L1CaloTauCorr );
}

iEvent.put( result , "CalibratedTaus");
 
}


double L1CaloTauCorrectionsProducer::GetCorrectionFactor(double Et, double Eta){  

  double corrFactor = 0;
  double myEtaMap[] = {-2.3, -2.1, -1.9, -1.7, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, +0.1, +0.3, +0.5, +0.7, +0.9, +1.1, +1.3, +1.5, +1.7, +1.9, +2.1, +2.3};
  int arrayIndex    = -1;
  arrayIndex        = L1CaloTauCorrectionsProducer::FindNearestIndex( 24, myEtaMap, Eta );

  /// Using "PrivateProduction2014" produced by Emanuelle (tmp solution)
  double Et_LEQ20     [] = {1.71, 1.82, 1.64, 1.46, 1.07, 1.31, 1.37, 1.37, 1.31, 1.35, 1.54, 1.52, 1.42, 1.31, 1.26, 1.36, 1.51, 1.22, 1.15, 1.14, 1.39, 1.42, 1.34, 1.31};
  double Et_G20_LEQ40 [] = {1.10, 1.14, 1.19, 1.12, 0.89, 0.94, 1.07, 1.06, 0.99, 1.10, 1.08, 1.14, 1.16, 1.09, 1.09, 1.00, 1.02, 0.99, 0.92, 0.90, 1.15, 1.23, 1.11, 1.13};
  double Et_G40_LEQ60 [] = {1.01, 0.99, 1.05, 0.99, 0.84, 0.81, 0.87, 0.87, 0.93, 0.95, 0.98, 0.99, 0.96, 1.00, 0.96, 0.93, 0.89, 0.86, 0.86, 0.81, 1.00, 1.06, 1.01, 1.00};
  double Et_G60_LEQ80 [] = {0.98, 0.90, 0.96, 0.91, 0.75, 0.76, 0.77, 0.86, 0.84, 0.90, 0.87, 0.93, 0.94, 0.87, 0.88, 0.86, 0.78, 0.77, 0.76, 0.76, 0.88, 1.02, 0.89, 1.00};
  double Et_G80       [] = {0.91, 0.91, 0.66, 0.66, 0.65, 0.68, 0.71, 0.74, 0.75, 0.83, 0.77, 0.77, 0.77, 0.74, 0.72, 0.77, 0.76, 0.76, 0.71, 0.65, 0.71, 0.71, 0.75, 0.91};
  
  /// Using "TTI2023Upg14D" official producation (for Technical Proposal), dataset  "SingleTauOneProng" 
  // double Et_LEQ20     [] = {2.26, 2.31, 1.78, 1.44, 1.12, 1.13, 1.17, 1.16, 1.23, 1.27, 1.29, 1.34, 1.26, 1.3 , 1.26, 1.28, 1.17, 1.13, 1.15, 1.16, 1.4 , 1.51, 2.13, 2.1 };
  // double Et_G20_LEQ40 [] = {1.08, 1.04, 1.15, 1.09, 0.84, 0.88, 0.93, 0.97, 1.01, 1.03, 1.05, 1.08, 1.07, 1.07, 1.02, 1.0 , 0.98, 0.92, 0.9 , 0.84, 1.08, 1.13, 1.04, 1.08};
  // double Et_G40_LEQ60 [] = {1.05, 1.0 , 1.04, 1.01, 0.81, 0.81, 0.84, 0.89, 0.93, 0.95, 0.98, 1.0 , 1.0 , 0.99, 0.96, 0.91, 0.89, 0.85, 0.83, 0.77, 1.01, 1.06, 1.02, 1.07};
  // double Et_G60_LEQ80 [] = {1.01, 0.97, 0.99, 0.96, 0.74, 0.79, 0.79, 0.83, 0.88, 0.91, 0.95, 0.97, 0.95, 0.91, 0.89, 0.87, 0.84, 0.82, 0.77, 0.74, 0.95, 0.99, 0.97, 1.03};
  // double Et_G80       [] = {0.93, 0.86, 0.91, 0.86, 0.67, 0.69, 0.73, 0.78, 0.79, 0.83, 0.85, 0.87, 0.87, 0.85, 0.84, 0.81, 0.77, 0.74, 0.72, 0.67, 0.87, 0.89, 0.86, 0.93};

  // Determine the calibration correction factor
  if ( Et <= 20.0){  corrFactor = Et_LEQ20[arrayIndex]; }
  else if ( ( 20.0 < Et ) && ( Et <= 40.0) ){ corrFactor = Et_G20_LEQ40[arrayIndex]; }
  else if ( ( 40.0 < Et ) && ( Et <= 60.0) ){ corrFactor = Et_G40_LEQ60[arrayIndex]; }
  else if ( ( 60.0 < Et ) && ( Et <= 80.0) ){ corrFactor = Et_G60_LEQ80[arrayIndex]; }
  else if ( 80.0 < Et ) corrFactor = Et_G80[arrayIndex]; 
  else {  throw cms::Exception("Logic") << "Could not find correction factor for L1 CaloTau with Et = " <<  Et << " and Eta = " << Eta  << std::endl;}

  // std::cout << "*** Et = " << Et << ", Eta = " << Eta << ", ArrayIndex  = " << arrayIndex << ", CorrFactor = " << corrFactor << std::endl;
  return corrFactor;

  }


int L1CaloTauCorrectionsProducer::FindNearestIndex(const unsigned int arraySize, double myArray[], double myEtaValue){
  
  
  // Variable declaration/initialisation
  int index   = -1;
  int counter = 0;
  std::vector<double> v_eta;
  v_eta.reserve(arraySize);

  // Convert array to vector to simplify things
  for(unsigned int i = 0; i< arraySize; i++){ v_eta.push_back( myArray[i]); }

  // Loop over all eta values and find the array index with a value closest to a chosen Eta value
  std::vector<double>::iterator it_eta = v_eta.begin();
  double best_diff = 9999.9;
  for( it_eta = v_eta.begin(); it_eta != v_eta.end(); it_eta++){

    double diff     = std::abs( *(it_eta) - myEtaValue );
    if ( diff < best_diff ){ 
      index     = counter;
      best_diff = diff;
    }
    counter++;
    
  }

  return index;

}



// ------------ method called once each job just before starting event loop  ------------
void 
L1CaloTauCorrectionsProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CaloTauCorrectionsProducer::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1CaloTauCorrectionsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//The following says we do not know what parameters are allowed so do no validation
// Please change this to state exactly what you do use, even if it is no parameters
edm::ParameterSetDescription desc;
desc.setUnknown();
descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloTauCorrectionsProducer);
