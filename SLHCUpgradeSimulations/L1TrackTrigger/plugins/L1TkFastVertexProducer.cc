// -*- C++ -*-
//
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// $Id$
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"


//#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"


////////////////////////////
// HepMC products
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"


#include "TH1F.h"



//
// class declaration
//

class L1TkFastVertexProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkFastVertexProducer(const edm::ParameterSet&);
      ~L1TkFastVertexProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------

        edm::InputTag L1TrackInputTag;

	float ZMAX;	// in cm
	float DeltaZ;	// in cm
	float CHI2MAX;
	float PTMINTRA ; 	// in GeV

	float PTMAX;	// in GeV, saturation / truncation value
	int HighPtTracks;	// saturate or truncate

	int nStubsmin ;		// minimum number of stubs 
	int nStubsPSmin ;	// minimum number of stubs in PS modules 

        int nBinning;   // number of bins used in the temp histogram

        bool MonteCarloVertex;   //
        //const StackedTrackerGeometry*                   theStackedGeometry;

	bool doPtComp ;
        bool doTightChi2 ;
  
        int WEIGHT; // weight (power) of pT 0 , 1, 2

	TH1F* htmp;
        TH1F* htmp_weight;


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
L1TkFastVertexProducer::L1TkFastVertexProducer(const edm::ParameterSet& iConfig)
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
 
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
   //now do what ever other initialization is needed
  
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

  ZMAX = (float)iConfig.getParameter<double>("ZMAX");
  CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");

  PTMAX = (float)iConfig.getParameter<double>("PTMAX");
  HighPtTracks = iConfig.getParameter<int>("HighPtTracks");

  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");
  nBinning = iConfig.getParameter<int>("nBinning");

  MonteCarloVertex = iConfig.getParameter<bool>("MonteCarloVertex");
  doPtComp = iConfig.getParameter<bool>("doPtComp");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");

  WEIGHT = iConfig.getParameter<int>("WEIGHT");

 //int nbins = 300;
 //float xmin = -15;
 //float xmax = +15;

 //int nbins = 600;
 //float xmin = -15;
 //float xmax = +15;

 //int nbins = 600;
 int nbins = nBinning ; // should be odd
 float xmin = -30 ;
 float xmax = +30 ;

  htmp = new TH1F("htmp",";z (cm); Tracks",nbins,xmin,xmax);
  htmp_weight = new TH1F("htmp_weight",";z (cm); Tracks",nbins,xmin,xmax);



  produces<L1TkPrimaryVertexCollection>();

}


L1TkFastVertexProducer::~L1TkFastVertexProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TkFastVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
 
 std::auto_ptr<L1TkPrimaryVertexCollection> result(new L1TkPrimaryVertexCollection);

 /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  const StackedTrackerGeometry* theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry
  //if (theStackedGeometry == 0) cout << " theStackedGeometry = 0 " << endl;      // for compil when not used...
        

 htmp -> Reset();
 htmp_weight -> Reset();


    // ----------------------------------------------------------------------

 if (MonteCarloVertex) {

        // MC info  ... retrieve the zvertex            
  edm::Handle<edm::HepMCProduct> HepMCEvt;
  iEvent.getByLabel("generator",HepMCEvt);
  
  edm::Handle< vector<reco::GenParticle> > GenParticleHandle;
  iEvent.getByLabel("genParticles","",GenParticleHandle);

     const double mm=0.1;
     float zvtx_gen = -999;

 
     if ( HepMCEvt.isValid() ) {
                // using HepMCEvt

     const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
     for ( HepMC::GenEvent::vertex_const_iterator ivertex = MCEvt->vertices_begin(); ivertex != MCEvt->vertices_end(); ++ivertex )
         {
             bool hasParentVertex = false;

             // Loop over the parents looking to see if they are coming from a production vertex
             for (
                 HepMC::GenVertex::particle_iterator iparent = (*ivertex)->particles_begin(HepMC::parents);
                 iparent != (*ivertex)->particles_end(HepMC::parents);
                 ++iparent
             )
                 if ( (*iparent)->production_vertex() )
                 {
                     hasParentVertex = true;
                     break;
                }
  
             // Reject those vertices with parent vertices
             if (hasParentVertex) continue;             
             // Get the position of the vertex
             HepMC::FourVector pos = (*ivertex)->position();
             zvtx_gen = pos.z()*mm;
             break;  // there should be one single primary vertex
          }  // end loop over gen vertices

	}
     else if (GenParticleHandle.isValid() ) {
        vector<reco::GenParticle>::const_iterator genpartIter ;
        for (genpartIter = GenParticleHandle->begin(); genpartIter != GenParticleHandle->end(); ++genpartIter) {
           int status = genpartIter -> status() ;
	   if (status != 3) continue;
	   if ( genpartIter -> numberOfMothers() == 0) continue;   // the incoming hadrons
	   float part_zvertex = genpartIter -> vz() ;
	   zvtx_gen = part_zvertex ;
	   break;	// 
	}
     }
     else {
	edm::LogError("L1TkFastVertexProducer")
 	  << "\nerror: try to retrieve the MC vertex (MonteCarloVertex = True) "
          << "\nbut the input file contains neither edm::HepMCProduct>  nor vector<reco::GenParticle>. Exit"
          << std::endl;
     }

     //     std::cout<<zvtx_gen<<endl;

     L1TkPrimaryVertex genvtx( zvtx_gen, -999.); 
     
     result -> push_back( genvtx );
     iEvent.put( result);
     return;
 }

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);   


 if( !L1TkTrackHandle.isValid() )
        {
          LogError("L1TkFastVertexProducer")
            << "\nWarning: L1TkTrackCollection with " << L1TrackInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
 	    return;
        }



  L1TkTrackCollectionType::const_iterator trackIter;
  for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {

    float z  = trackIter->getPOCA().z();
    float chi2 = trackIter->getChi2();
    float pt = trackIter->getMomentum().perp();
    float eta  = trackIter ->getMomentum().eta();

    //..............................................................    
    float wt = pow(pt,WEIGHT); // calculating the weight for tks in as pt^0,pt^1 or pt^2 based on WEIGHT


    if (fabs(z) > ZMAX ) continue;
    if (chi2 > CHI2MAX) continue;
    if (pt < PTMINTRA) continue;

    // saturation or truncation :
   if ( PTMAX > 0 && pt > PTMAX ) {
	if (HighPtTracks == 0)  continue;	// ignore this track
	if (HighPtTracks == 1)  pt = PTMAX;	// saturate
   }
  

        // get the number of stubs and the number of stubs in PS layers
    float nPS = 0.;     // number of stubs in PS modules
    float nstubs = 0;

      // get pointers to stubs associated to the L1 track
	std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

      int tmp_trk_nstub = (int) theStubs.size();
      if ( tmp_trk_nstub < 0) {
        std::cout << " ... could not retrieve the vector of stubs in L1TkFastVertexProducer::SumPtVertex " << std::endl;
        continue;
      }


      // loop over the stubs
      for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
           nstubs ++;
           StackedTrackerDetId detIdStub( theStubs.at(istub)->getDetId() );
           bool isPS = theStackedGeometry -> isPSModule( detIdStub );
           if (isPS) nPS ++;
       } // end loop over stubs
        if (nstubs < nStubsmin) continue;
        if (nPS < nStubsPSmin) continue;



	// quality cuts from Louise S, based on the pt-stub compatibility (June 20, 2014)
    int trk_nstub  = (int) trackIter ->getStubRefs().size();
    float chi2dof = chi2 / (2*trk_nstub-4);

     if (doPtComp) {
      float trk_consistency = trackIter ->getStubPtConsistency();
      //if (trk_nstub < 4) continue;	// done earlier
      //if (chi2 > 100.0) continue;	// done earlier
      if (trk_nstub == 4) {
    	if (fabs(eta)<2.2 && trk_consistency>10) continue;
    	else if (fabs(eta)>2.2 && chi2dof>5.0) continue;
      }
    }
    if (doTightChi2) {
          if (pt>10.0 && chi2dof>5.0) continue;
    }

     htmp -> Fill( z );
     htmp_weight -> Fill( z, wt );// changed from "pt" to "wt" which is some power of pt (0,1 or 2)
  

     
  } // end loop over tracks
  

/*
  int binmax = htmp -> GetMaximumBin();
  float zvtx = htmp -> GetBinCenter( binmax );
  binmax = htmp_weight -> GetMaximumBin();
  float zvtx_weight = htmp_weight -> GetBinCenter( binmax );

   L1TkPrimaryVertex vtx1( zvtx, zvtx_gen);
   L1TkPrimaryVertex vtx2( zvtx_weight, zvtx_gen);
*/


        // sliding windows... maximize bin i + i-1  + i+1
        
  float zvtx_sliding = -999;
  float sigma_max = -999;
  int nb = htmp -> GetNbinsX();
  for (int i=2; i <= nb-1; i++) {
     float a0 = htmp -> GetBinContent(i-1);
     float a1 = htmp -> GetBinContent(i);
     float a2 = htmp -> GetBinContent(i+1);
     float sigma = a0 + a1 + a2;
     if (sigma > sigma_max) {
        sigma_max = sigma;
        float z0 = htmp -> GetBinCenter(i-1);
        float z1 = htmp -> GetBinCenter(i);
        float z2 = htmp -> GetBinCenter(i+1);
        zvtx_sliding =  ( a0 * z0 + a1 * z1 + a2 * z2 ) / sigma;
     }  
  } 
    
  //L1TkPrimaryVertex vtx3( zvtx_sliding, zvtx_gen);
      
  zvtx_sliding = -999;
  sigma_max = -999;
  for (int i=2; i <= nb-1; i++) {
     float a0 = htmp_weight -> GetBinContent(i-1);
     float a1 = htmp_weight -> GetBinContent(i); 
     float a2 = htmp_weight -> GetBinContent(i+1);
     float sigma = a0 + a1 + a2;
     if (sigma > sigma_max) {
        sigma_max = sigma;
        float z0 = htmp_weight -> GetBinCenter(i-1);
        float z1 = htmp_weight -> GetBinCenter(i);
        float z2 = htmp_weight -> GetBinCenter(i+1);
        zvtx_sliding =  ( a0 * z0 + a1 * z1 + a2 * z2 ) / sigma;
     }
  }
  //  cout<<zvtx_sliding<<"\t"<< sigma_max<<endl;
 //L1TkPrimaryVertex vtx4( zvtx_sliding, zvtx_gen);
 L1TkPrimaryVertex vtx4( zvtx_sliding, sigma_max);


/*
 int NTRAMIN = 2;
 TH1F* htmp_weight_cleaned = (TH1F*)htmp_weight -> Clone();
 for (int i=0; i<= nb; i++) { 
   float val = htmp_weight -> GetBinContent(i);
   float Ntracks = htmp -> GetBinContent(i);
   if ( Ntracks >= NTRAMIN) {  
        htmp_weight_cleaned -> SetBinContent(i, val);
   }
   else {
        htmp_weight_cleaned -> SetBinContent(i, 0.);
   }
 }
  
  binmax = htmp_weight_cleaned -> GetMaximumBin();
  float zvtx_weight_cleaned = htmp_weight_cleaned -> GetBinCenter( binmax );
  L1TkPrimaryVertex vtx5( zvtx_weight_cleaned, zvtx_gen);
*/

 result -> push_back( vtx4 );

/*
 result -> push_back( vtx1 );
 result -> push_back( vtx2 );
 result -> push_back( vtx3 );
 result -> push_back( vtx5 );
*/

 iEvent.put( result);
}



// ------------ method called once each job just before starting event loop  ------------
void 
L1TkFastVertexProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TkFastVertexProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TkFastVertexProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{

}
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkFastVertexProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkFastVertexProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkFastVertexProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkFastVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkFastVertexProducer);
