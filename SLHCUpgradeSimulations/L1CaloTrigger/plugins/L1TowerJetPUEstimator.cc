// -*- C++ -*-
//
// Package:    L1TowerJetPUEstimator
// Class:      L1TowerJetPUEstimator
// 
/**\class CalibTowerJetCollection L1TowerJetPUEstimator.cc MVACALIB/L1TowerJetPUEstimator/src/L1TowerJetPUEstimator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Robyn Elizabeth Lucas,510 1-002,+41227673823,
//         Created:  Mon Nov 19 10:20:06 CET 2012
// $Id: L1TowerJetPUEstimator.cc,v 1.1 2013/03/21 17:23:28 rlucas Exp $
//
//


// system include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TLorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include <iostream>
#include <fstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"
//
// class declaration
//


using namespace l1slhc;
using namespace edm;
using namespace std;
using namespace reco;
using namespace l1extra;


bool myfunction (double i,double j) { return (i>j); }

bool sortTLorentz (TLorentzVector i,TLorentzVector j) { return ( i.Pt()>j.Pt() ); }


class L1TowerJetPUEstimator : public edm::EDProducer {
   public:
      explicit L1TowerJetPUEstimator(const edm::ParameterSet&);
      ~L1TowerJetPUEstimator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
    
      double get_rho(double L1rho);

    //fwd calibration: V ROUGH (only to L1extra particles)
//       double rough_ptcal(double pt);
    
      double Median(vector<double> aVec);

      // ----------member data ---------------------------
      ParameterSet conf_;
    
      ifstream inrhodata;
    
      vector < pair<double, double> > rho_cal_vec;

      edm::FileInPath inRhoData_edm;
   
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

L1TowerJetPUEstimator::L1TowerJetPUEstimator(const edm::ParameterSet& iConfig):
conf_(iConfig)
{

    produces<double>("Rho");

}


L1TowerJetPUEstimator::~L1TowerJetPUEstimator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}




// ------------ method called to produce the data  ------------
void
L1TowerJetPUEstimator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    bool evValid =true;
    double outrho(0);
    auto_ptr<double> outRho(new double());

    edm::Handle<L1TowerJetCollection > UnCalibCen;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("FilteredCircle8"), UnCalibCen);
    if(!UnCalibCen.isValid()){evValid=false;}

    if( !evValid ) {
      //edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("FilteredCircle8") << "," << conf_.getParameter<edm::InputTag>("FilteredFwdCircle8") << std::endl; 
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("FilteredCircle8") << std::endl; 
    }
    else{

      //produce calibrated rho collection

      double areaPerJet(9999);
      int count(0);
      vector<double> Jet2Energies;
      for (L1TowerJetCollection::const_iterator il1 = UnCalibCen->begin();
        il1!= UnCalibCen->end() ; ++il1 ){
        if( abs(il1->p4().eta() )>3) continue;
        if(count>1) {
          Jet2Energies.push_back(il1->p4().Pt());   
          //cout<<"jet energy: "<< il1->p4().Pt() <<endl;
        }
        count++;
        areaPerJet = il1->JetArea()* (0.087 * 0.087) ;
      }
      double raw_rho2 = ( Median( Jet2Energies ) / areaPerJet );
      
      //apply calibration to the raw rho: should we do this at this stage?
      //not sure of the effect on high PU data

      ///////////////////////////////////////////////////
      //              SET VALUE OF RHO 
      ///////////////////////////////////////////////////
      outrho=raw_rho2;
      
      *outRho = outrho;

      iEvent.put(outRho,"Rho");

    } //valid event
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TowerJetPUEstimator::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TowerJetPUEstimator::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1TowerJetPUEstimator::beginRun(edm::Run&, edm::EventSetup const&)
{    
}

// ------------ method called when ending the processing of a run  ------------
void 
L1TowerJetPUEstimator::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TowerJetPUEstimator::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TowerJetPUEstimator::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TowerJetPUEstimator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//
// member functions
//
double L1TowerJetPUEstimator::Median( vector<double> aVec){
    sort( aVec.begin(), aVec.end() );
    double median(0);
    int size = aVec.size();
    if(size ==0){
        median = 0;
    }
    else if(size==1){
        median = aVec[size-1];
    }
    else if( size%2 == 0 ){
        median = ( aVec[ (size/2)-1  ] + aVec[ (size /2) ] )/2;
    }else{
        median = aVec [ double( (size/2) ) +0.5 ];
    }
    return median;
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1TowerJetPUEstimator);
