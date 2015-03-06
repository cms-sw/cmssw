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
// $Id: L1TowerJetPUEstimator.cc,v 1.3 2013/05/16 17:35:30 mbaber Exp $
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

      // Local rho eta region boundaries
      vector < double > localRhoEtaDiv;
      // Minimum jets in region for the calulation of local rho
      unsigned int minimumLocalJets;
  
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

  std::cout << "\n\n----------------------------------------\nBegin: L1TowerJetPUEstimator\n----------------------------------------\n\n";
    
    // look up tables
    inRhoData_edm   = iConfig.getParameter<edm::FileInPath> ("inRhodata_file");
    // determine whether to perform the rho calibration
    useRhoCalib     = iConfig.getParameter< bool >("UseRhoCalibration");
    // number of jets to exclude from the median calculation for rho, subtracting 1 to transform to a jet index
    skipJetsIndex   = iConfig.getParameter< unsigned int >("numberOfSkippedJets") - 1;
    // Divisions of the eta regions overwhich to calculate local rhos
    localRhoEtaDiv  = iConfig.getParameter< vector< double > >("LocalRhoEtaDivisions");
    // Minimum number of jets required in an eta region to perform a measurement of rho
    minimumLocalJets = iConfig.getParameter< unsigned int >("LocalRhoMinJetsInRegion");


    // Ensure the divisions are in ascending order of eta
    sort (localRhoEtaDiv.begin(), localRhoEtaDiv.end());


    produces<bool>("RhoCalibrated");
    produces<double>("Rho");
    produces< vector< double > >("LocalRho"); 
    produces< vector< double > >("LocalRhoEtaBoundaries");


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
    // Local rho in corresponding bin
    auto_ptr< vector< double > > localRhoCollection( new vector< double >() );


    // Incredibly inefficienct - This should just be read from the config file once in each module that requires it
    //
    //
    // Lower eta edge of rho bin, final bin for upper edge of entire range
    //    localEtaBoundaries = auto_ptr< vector< double > >( new vector< double > (localRhoEtaDiv) ); 
    auto_ptr< vector< double > > localEtaBoundaries = auto_ptr< vector< double > >( new vector< double > () ); 
    
    for (unsigned int iSlice = 0; iSlice < localRhoEtaDiv.size(); iSlice++){
      double eta  = localRhoEtaDiv[iSlice];
      localEtaBoundaries->push_back( eta );
    }
    //
    //
    //


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


      // Produce global rho collection
      unsigned int jetIndex(0);

      // Jet energy densities inside the entire calorimeter (global) and local regions
      vector <double> jetPtAreaRatioGlobal;
      vector < vector<double> > jetPtAreaRatioLocal;

      // Resize the local array to fit the required number of regions
      jetPtAreaRatioLocal.resize( localRhoEtaDiv.size() - 1 );


      for (L1TowerJetCollection::const_iterator il1 = UnCalibCen->begin(); il1!= UnCalibCen->end(); ++il1 ){



	// Restrict to jets in barrel and endcap. This will need to be removed when HF jets are included.
	double weightedEta = il1->WeightedEta();
        if( fabs( weightedEta ) > 3.) continue;


	// Skip the specified number of jets in the calculation of the median for rho
        if( jetIndex > skipJetsIndex ){


	  // **********************************************************************
	  // *                             Global rho                             *
	  // **********************************************************************

	  // Store the global jet energy density
	  double ptAreaRatio = il1->Pt() / il1->JetRealArea();
          jetPtAreaRatioGlobal.push_back( ptAreaRatio );   


	  // **********************************************************************
	  // *                              Local rho                             *
	  // **********************************************************************

	  // Determine in which region the jet resides
	  for (unsigned int iSlice = 0;iSlice < localRhoEtaDiv.size() - 1; iSlice++){

	    // Get the current eta slice range
	    double etaLow  = localRhoEtaDiv[iSlice];
	    double etaHigh = localRhoEtaDiv[iSlice + 1];

	    // Store the jet in the respective eta region
	    if ( (weightedEta >= etaLow) && (weightedEta < etaHigh) ){

	      // Fill the jet in respective vector for calculating local rho (lower edge)
	      jetPtAreaRatioLocal[iSlice].push_back( ptAreaRatio );
	      
	      //	      std::cout << "pT = " << il1->Pt() << "\tArea = " << il1->JetRealArea() << "\tEta = " << weightedEta << "\tetaRange = (" << etaLow << ", " << etaHigh << ")\n";

	    }


	  }


	}
        jetIndex++;
      }

      // Calculate global rho, the median jet energy density
      double globalRho = Median(jetPtAreaRatioGlobal);


      // Calculate local rhos for each eta bin
      for (unsigned int iSlice = 0; iSlice < jetPtAreaRatioLocal.size(); iSlice++){

	double localRho;

	// Check whether enough jets are present to perform local PU subtraction
	if (jetPtAreaRatioLocal[iSlice].size() >= minimumLocalJets ){
	  // Calculate the rho local to the current bin
	  localRho = Median( jetPtAreaRatioLocal[iSlice] );
	}
	else{ // Insufficient statistics to obtain a useful measure of rho of the region
	  localRho = 0;
	}



	// Store the local rho
	localRhoCollection->push_back(localRho);
	//localEtaBoundaries->push_back( etaLow );

      }


      // Determine whether to apply the rho calibration (Only to global rho)
      if ( useRhoCalib ){
	
	//apply calibration to the raw rho: should we do this at this stage? 
	//not sure of the effect on high PU data
	double cal_rhoL1 = globalRho * get_rho(globalRho);
	
	// return calibrated rho
	outrho = cal_rhoL1;

      }
      else{
	
	// return uncalibrated rho
	outrho = globalRho;
      }







      *outRho = outrho;

      // Return whether rho calibration was applied (global only)
      iEvent.put(useCalib,"RhoCalibrated");

      // Return global rho
      iEvent.put(outRho,"Rho");
      // Return local rho and the eta regions utilised
      iEvent.put(localRhoCollection,"LocalRho");
      iEvent.put(localEtaBoundaries,"LocalRhoEtaBoundaries");


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
