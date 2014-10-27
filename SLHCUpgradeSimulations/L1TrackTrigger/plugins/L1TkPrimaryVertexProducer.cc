// -*- C++ -*-
//
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// $Id$
//
//

// -------------------------------------------------------------------------------------------------------
//
//	********  OLD CODE   ********
//
//	********  The latest producer for the primary vertex is  L1TkFastVertexProducer.cc      ********
//
// --------------------------------------------------------------------------------------------------------



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


//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
//#include "SimDataFormats/SLHC/interface/slhcevent.hh"
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


#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"


//
// class declaration
//

class L1TkPrimaryVertexProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkPrimaryVertexProducer(const edm::ParameterSet&);
      ~L1TkPrimaryVertexProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


      float MaxPtVertex(const edm::Handle<L1TkTrackCollectionType> & L1TkTrackHandle,
                float& sum,
                int nmin, int nPSmin, float ptmin, int imode,
		const StackedTrackerGeometry* theStackedGeometry) ;

      float SumPtVertex(const edm::Handle<L1TkTrackCollectionType> & L1TkTrackHandle,
                float z, int nmin, int nPSmin, float ptmin, int imode,
		const StackedTrackerGeometry* theStackedGeometry ) ;


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

	int nStubsmin ;		// minimum number of stubs 
	int nStubsPSmin ;	// minimum number of stubs in PS modules 
	bool SumPtSquared;

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
L1TkPrimaryVertexProducer::L1TkPrimaryVertexProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   //now do what ever other initialization is needed
  
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

  ZMAX = (float)iConfig.getParameter<double>("ZMAX");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");

  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");

  SumPtSquared = iConfig.getParameter<bool>("SumPtSquared");

  produces<L1TkPrimaryVertexCollection>();

}


L1TkPrimaryVertexProducer::~L1TkPrimaryVertexProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TkPrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

 std::auto_ptr<L1TkPrimaryVertexCollection> result(new L1TkPrimaryVertexCollection);

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  //const TrackerGeometry*                                       theGeometry;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;
  
  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  //theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); 
  
  ////////////////////////
  // GET MAGNETIC FIELD //
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  if ( mMagneticFieldStrength < 0) std::cout << "mMagneticFieldStrength < 0 " << std::endl;  // for compil when not used



  edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
  //iEvent.getByLabel("L1Tracks","Level1TkTracks",L1TkTrackHandle);
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);   


 if( !L1TkTrackHandle.isValid() )
        {
          LogError("L1TkPrimaryVertexProducer")
            << "\nWarning: L1TkTrackCollection with " << L1TrackInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
 	    return;
        }



   float sum1 = -999;
   int nmin = nStubsmin;
   int nPSmin = nStubsPSmin ;
   float ptmin = PTMINTRA ;
   int imode = 2;	// max(Sum PT2)
   if (! SumPtSquared)  imode = 1;   // max(Sum PT)

   float z1 = MaxPtVertex( L1TkTrackHandle, sum1, nmin, nPSmin, ptmin, imode, theStackedGeometry );
   L1TkPrimaryVertex vtx1( z1, sum1 );

 result -> push_back( vtx1 );

 iEvent.put( result);
}


float L1TkPrimaryVertexProducer::MaxPtVertex(const edm::Handle<L1TkTrackCollectionType> & L1TkTrackHandle,
 		float& Sum,
		int nmin, int nPSmin, float ptmin, int imode,
		const StackedTrackerGeometry*                   theStackedGeometry) {
        // return the zvtx corresponding to the max(SumPT)
        // of tracks with at least nPSmin stubs in PS modules
   
      float sumMax = 0;
      float zvtxmax = -999;
      int nIter = (int)(ZMAX * 10. * 2.) ;
      for (int itest = 0; itest <= nIter; itest ++) {
	
        //float z = -100 + itest;         // z in mm
	float z = -ZMAX * 10 + itest ;  	// z in mm
        z = z/10.  ;   // z in cm
        float sum = SumPtVertex(L1TkTrackHandle, z, nmin, nPSmin, ptmin, imode, theStackedGeometry);
        if (sumMax >0 && sum == sumMax) {
          //cout << " Note: Several vertices have the same sum " << zvtxmax << " " << z << " " << sumMax << endl;
        }
   
        if (sum > sumMax) {
           sumMax = sum;
           zvtxmax = z;
        }
       }  // end loop over tested z 
   
 Sum = sumMax;
 return zvtxmax;
}  


float L1TkPrimaryVertexProducer::SumPtVertex(const edm::Handle<L1TkTrackCollectionType> & L1TkTrackHandle,
		float z, int nmin, int nPSmin, float ptmin, int imode,
		const StackedTrackerGeometry*                   theStackedGeometry) {

        // sumPT of tracks with >= nPSmin stubs in PS modules
        // z in cm
 float sumpt = 0;


  L1TkTrackCollectionType::const_iterator trackIter;

  for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {

    float pt = trackIter->getMomentum().perp();
    float chi2 = trackIter->getChi2();
    float ztr  = trackIter->getPOCA().z();

    if (pt < ptmin) continue;
    if (fabs(ztr) > ZMAX ) continue;
    if (chi2 > CHI2MAX) continue;
    if ( fabs(ztr - z) > DeltaZ) continue;   // eg DeltaZ = 1 mm


	// get the number of stubs and the number of stubs in PS layers
    float nPS = 0.;     // number of stubs in PS modules
    float nstubs = 0;

      // get pointers to stubs associated to the L1 track
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

      int tmp_trk_nstub = (int) theStubs.size();
      if ( tmp_trk_nstub < 0) {
	std::cout << " ... could not retrieve the vector of stubs in L1TkPrimaryVertexProducer::SumPtVertex " << std::endl;
	continue;
      }

      // loop over the stubs
      for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
        //bool genuine = theStubs.at(istub)->isGenuine();
        //if (genuine) {
           nstubs ++;
           StackedTrackerDetId detIdStub( theStubs.at(istub)->getDetId() );
           bool isPS = theStackedGeometry -> isPSModule( detIdStub );
           //if (isPS) cout << " this is a stub in a PS module " << endl;
           if (isPS) nPS ++;
	//} // endif genuine
       } // end loop over stubs

        if (imode == 1 || imode == 2 ) {
            if (nPS < nPSmin) continue;
        }
	if ( nstubs < nmin) continue;

        if (imode == 2) sumpt += pt*pt;
        if (imode == 1) sumpt += pt;

  } // end loop over the tracks

 return sumpt;

}



// ------------ method called once each job just before starting event loop  ------------
void 
L1TkPrimaryVertexProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TkPrimaryVertexProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TkPrimaryVertexProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{


}
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkPrimaryVertexProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkPrimaryVertexProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkPrimaryVertexProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkPrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkPrimaryVertexProducer);
