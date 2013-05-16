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
// Modifications  :  Mark Baber Imperial College, London
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
      
      // Determines the median value of a vector of doubles
      double Median(vector<double> aVec);


      // ----------member data ---------------------------
      ParameterSet conf_;
    
      ifstream inrhodata;
    
      vector < pair<double, double> > rho_cal_vec;

      edm::FileInPath inRhoData_edm;
   


      // Determines whether to calibrate rho to offline rho 
      bool useRhoCalib;
      // Inclusive upper limit of the jet indexes to exclude from the median calculation for rho from the ordered jet list
      // e.g. skipJetsIndex = 2  => Skip first three jets (indices = 0, 1, 2)
      unsigned int skipJetsIndex;

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


    
    // look up tables
    inRhoData_edm = iConfig.getParameter<edm::FileInPath> ("inRhodata_file");
    // determine whether to perform the rho calibration
    useRhoCalib   = iConfig.getParameter< bool >("UseRhoCalibration");
    // number of jets to exclude from the median calculation for rho, subtracting 1 to transform to a jet index
    skipJetsIndex = iConfig.getParameter< unsigned int >("numberOfSkippedJets") - 1;

    produces<double>("Rho");
    produces<bool>("RhoCalibrated");
      
}


L1TowerJetPUEstimator::~L1TowerJetPUEstimator()
{
}




// ------------ method called to produce the data  ------------
void
L1TowerJetPUEstimator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    bool evValid = true;
    double outrho(0);
    auto_ptr<double> outRho(new double());
    auto_ptr<bool>   useCalib(new bool());

    // Store whether rho calibration was applied
    *useCalib = useRhoCalib;

    edm::Handle<L1TowerJetCollection > UnCalibCen;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("FilteredCircle8"), UnCalibCen);
    if(!UnCalibCen.isValid()){evValid=false;}



    if( !evValid ) {
      // ???? Surely this should throw an exception if collection is not present? ????
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("FilteredCircle8")      
					<< std::endl; 
    }
    else{

      //produce calibrated rho collection
      unsigned int jetIndex(0);

      vector<double> jetPtAreaRatio;
      for (L1TowerJetCollection::const_iterator il1 = UnCalibCen->begin(); il1!= UnCalibCen->end(); ++il1 ){
        if( fabs(il1->Eta() ) > 3) continue;

	// Skip the specified number of jets in the calculation of the median for rho
        if( jetIndex > skipJetsIndex ){
	  // Store the jet energy density
          jetPtAreaRatio.push_back( il1->Pt() / il1->JetRealArea() );   
	}
        jetIndex++;
      }

      // Calculate rho, the median jet energy density
      double raw_rho2 = Median(jetPtAreaRatio);

      // Determine whether to apply the rho calibration
      if ( useRhoCalib ){
	
	//apply calibration to the raw rho: should we do this at this stage? 
	//not sure of the effect on high PU data
	double cal_rhoL1 = raw_rho2 * get_rho(raw_rho2);
	
	// return calibrated rho
	outrho = cal_rhoL1;

      }
      else{
	
	// return uncalibrated rho
	outrho = raw_rho2;
      }

      *outRho = outrho;

      iEvent.put(outRho,"Rho");
      iEvent.put(useCalib,"RhoCalibrated");

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

  if (useRhoCalib){
    //read in calibration for rho lookup table
    inrhodata.open(inRhoData_edm.fullPath().c_str());
    if(!inrhodata) cerr << " unable to open rho lookup file. " << endl;

    //read into a vector
    pair<double, double> rho_cal;
    double L1rho_(9999), calFac_(9999);
    while ( !inrhodata.eof() ) { // keep reading until end-of-file
      // sets EOF flag if no value found
      inrhodata >> L1rho_ >> calFac_ ;
      
      rho_cal.first = L1rho_;
      rho_cal.second= calFac_;
      
      rho_cal_vec.push_back(rho_cal);
    }
    inrhodata.close();
    
    std::cout << "\nRead in Rho lookup table\n";
  }
  else{
    std::cout << "\nWARNING: Not performing rho calibration\n";
  }
   
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

  // Order vector collection
  sort( aVec.begin(), aVec.end() );

  double median(0);
  int size = aVec.size();
  int halfSize = size/2;
  if( size == 0 ){
    median = 0;
  }
  else if( size == 1 ){
    median = aVec[0];
  }
  else if( size%2 == 0 ){
    // Even number of entries, take average of the values around center
    median = ( aVec[ halfSize - 1 ] + aVec[ halfSize ] ) * 0.5;
  }
  else{
    // Odd number of entries, halfSize is central element
    median = aVec[ halfSize ];
  }
  
  return median;
}


      



double L1TowerJetPUEstimator::get_rho(double L1_rho)
{

  // Get the rho multiplication factor:
  // L1_rho * 2 gets the array index
  if(L1_rho <= 40.5){
    return rho_cal_vec[L1_rho*2].second;
  }
  else{ //for L1 rho > 40.5, calibration flattens out
    return 1.44576;
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TowerJetPUEstimator);
