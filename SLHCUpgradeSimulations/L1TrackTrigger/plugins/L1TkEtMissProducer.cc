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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/LorentzVector.h"

//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
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


#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticleFwd.h"


using namespace l1extra;

//
// class declaration
//

class L1TkEtMissProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkEtMissProducer(const edm::ParameterSet&);
      ~L1TkEtMissProducer();

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

        edm::InputTag L1VertexInputTag;
        edm::InputTag L1TrackInputTag;

	float ZMAX;	// in cm
	float DeltaZ;	// in cm
	float CHI2MAX;
	float PTMINTRA;	// in GeV
        int nStubsmin;
        int nStubsPSmin ;       // minimum number of stubs in PS modules 

	float PTMAX;	// in GeV
        int HighPtTracks;       // saturate or truncate

        bool doPtComp;
        bool doTightChi2;

        //const StackedTrackerGeometry*                   theStackedGeometry;

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
L1TkEtMissProducer::L1TkEtMissProducer(const edm::ParameterSet& iConfig)
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
  
  L1VertexInputTag = iConfig.getParameter<edm::InputTag>("L1VertexInputTag") ;
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");


  ZMAX = (float)iConfig.getParameter<double>("ZMAX");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");

  PTMAX = (float)iConfig.getParameter<double>("PTMAX");
  HighPtTracks = iConfig.getParameter<int>("HighPtTracks");
  doPtComp     = iConfig.getParameter<bool>("doPtComp");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");

  produces<L1TkEtMissParticleCollection>("MET");

}


L1TkEtMissProducer::~L1TkEtMissProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TkEtMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;
  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product();

 
 std::auto_ptr<L1TkEtMissParticleCollection> result(new L1TkEtMissParticleCollection);

 edm::Handle<L1TkPrimaryVertexCollection> L1VertexHandle;
 iEvent.getByLabel(L1VertexInputTag,L1VertexHandle);
 std::vector<L1TkPrimaryVertex>::const_iterator vtxIter;

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
 L1TkTrackCollectionType::const_iterator trackIter;


 if( !L1VertexHandle.isValid() )
        {
          LogError("L1TkEtMissProducer")
            << "\nWarning: L1TkPrimaryVertexCollection with " << L1VertexInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
	   return;
        }

 if( !L1TkTrackHandle.isValid() )
        {
          LogError("L1TkEtMissProducer")
            << "\nWarning: L1TkTrackCollectionType with " << L1TrackInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
           return;
        }


 int ivtx = 0;

 for (vtxIter = L1VertexHandle->begin(); vtxIter != L1VertexHandle->end(); ++vtxIter) {

    float zVTX = vtxIter -> getZvertex();
    edm::Ref< L1TkPrimaryVertexCollection > vtxRef( L1VertexHandle, ivtx );
    ivtx ++;

    float sumPx = 0;
    float sumPy = 0;
    float etTot = 0;

    double sumPx_PU = 0;
    double sumPy_PU = 0;
    double etTot_PU = 0;

  	for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {
 
    	    float pt = trackIter->getMomentum().perp();
    	    float eta = trackIter->getMomentum().eta();
    	    float chi2 = trackIter->getChi2();
    	    float ztr  = trackIter->getPOCA().z();

    	    if (pt < PTMINTRA) continue;
    	    if (fabs(ztr) > ZMAX ) continue;
    	    if (chi2 > CHI2MAX) continue;


	    float pt_rescale = 1;

	    if ( PTMAX > 0 && pt > PTMAX)  {
	        if (HighPtTracks == 0)  continue;	// ignore these very high PT tracks.
		if (HighPtTracks == 1)  {
			pt_rescale = PTMAX / pt;	// will be used to rescale px and py
			pt = PTMAX;     // saturate
		}
	    }

            int nstubs = 0;
	    float nPS = 0.;     // number of stubs in PS modules

	    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

            int tmp_trk_nstub = (int) theStubs.size();
            if ( tmp_trk_nstub < 0) continue;
            // loop over the stubs
            for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
                  nstubs ++;
           	  StackedTrackerDetId detIdStub( theStubs.at(istub)->getDetId() );
                  bool isPS = theStackedGeometry -> isPSModule( detIdStub );
           	  if (isPS) nPS ++;
            }
            if (nstubs < nStubsmin) continue;
            if (nPS < nStubsPSmin) continue;


	    ////_______
	    ////-------
	    float trk_consistency = trackIter ->getStubPtConsistency();
	    float chi2dof = chi2 / (2*nstubs-4);	

	    if(doPtComp) {
	      //	      if (trk_nstub < 4) continue;
	      //	      if (trk_chi2 > 100.0) continue;
	      if (nstubs == 4) {
		if (fabs(eta)<2.2 && trk_consistency>10) continue;
		else if (fabs(eta)>2.2 && chi2dof>5.0) continue;
	      }
	    }

	    if(doTightChi2) {
	      if(pt>10.0 && chi2dof>5.0 ) continue;
	    }

	    ////_______
	    ////-------



            if ( fabs(ztr - zVTX) <= DeltaZ) {   // eg DeltaZ = 1 mm

	    	sumPx += trackIter->getMomentum().x() * pt_rescale ;
	    	sumPy += trackIter->getMomentum().y() * pt_rescale ;
	    	etTot += pt ;
	    }
	    else   {	// PU sums
                sumPx_PU += trackIter->getMomentum().x() * pt_rescale ;
                sumPy_PU += trackIter->getMomentum().y() * pt_rescale ;
                etTot_PU += pt ;
	    }


    	} // end loop over tracks
     float et = sqrt( sumPx*sumPx + sumPy*sumPy );
     math::XYZTLorentzVector missingEt( -sumPx, -sumPy, 0, et);

     double etmiss_PU = sqrt( sumPx_PU*sumPx_PU + sumPy_PU*sumPy_PU );

     int ibx = 0;
     result -> push_back(  L1TkEtMissParticle( missingEt,
				 L1TkEtMissParticle::kMET,
				 etTot,
				 etmiss_PU,
				 etTot_PU,
				 vtxRef,
				 ibx )
		         ) ;


 } // end loop over vertices

 iEvent.put( result, "MET");
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TkEtMissProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TkEtMissProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TkEtMissProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{

/*
  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  //const TrackerGeometry*                                       theGeometry;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  //theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry
  if (theStackedGeometry == 0) cout << " theStackedGeometry = 0 " << endl;      // for compil when not used...
*/

}
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkEtMissProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkEtMissProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkEtMissProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkEtMissProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEtMissProducer);
