#ifndef L1TTRACK_DEGRADE_H
#define L1TTRACK_DEGRADE_H

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
// #include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
//#include "SimDataFormats/SLHC/interface/slhcevent.hh"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

//#include "SimDataFormats/SLHC/interface/L1TBarrel.hh"
//#include "SimDataFormats/SLHC/interface/L1TDisk.hh"
//#include "SimDataFormats/SLHC/interface/L1TStub.hh"


#include "TRandom.h"
#include "TMath.h"
#include <iostream>


using namespace edm;

class L1TrackDegrader : public edm::EDProducer
{ 
public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;

  /// Constructor/destructor
  explicit L1TrackDegrader(const edm::ParameterSet& iConfig);
  virtual ~L1TrackDegrader();

protected:

private:
         edm::InputTag L1TrackInputTag;

	 bool degradeZ0;
	 bool degradeMomentum;
	 int NsigmaPT ;	// smear the PT by N sigmas


	 TRandom ran;


  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


L1TrackDegrader::L1TrackDegrader(edm::ParameterSet const& iConfig) // :   config(iConfig)
{

   L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

   degradeZ0 = iConfig.getParameter<bool>("degradeZ0");
   degradeMomentum = iConfig.getParameter<bool>("degradeMomentum");
   NsigmaPT = iConfig.getParameter<int>("NsigmaPT");

   ran.SetSeed(0);

produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( "Level1TTTracks" ).setBranchAlias("Level1TTTracks");

}

L1TrackDegrader::~L1TrackDegrader()
{ 
  /// Insert here what you need to delete
  /// when you close the class instance
}


void L1TrackDegrader::beginRun(edm::Run& run, const edm::EventSetup& iSetup )
{
}


void L1TrackDegrader::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle); 
 L1TkTrackCollectionType::const_iterator trackIter;

  std::auto_ptr< L1TkTrackCollectionType > L1TkTracksForOutput( new L1TkTrackCollectionType );

  int Npar = 4;	// smear the 4-parameter tracks

        for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {

	   float Eta = fabs( trackIter->getMomentum(Npar).eta() );
	   //float Pt = trackIter->getMomentum().perp();
	   float z0 = trackIter->getPOCA(Npar).z(); 

           L1TkTrackType smearedTrack = *trackIter;

	   if ( degradeZ0 ) {

	      // hard-coding of the z-smearing based on Stefano's plots

	      float sigma= 0;

	      if (Eta >=  0.5 && Eta < 2) sigma = 0.168 * Eta - 0.08 ;
	      if (Eta >= 2 && Eta < 2.3) sigma = 0.256;

	      float deltaZ = ran.Gaus(0.,sigma);
	      float smeared_z = z0 + deltaZ ;

	      float x0 = trackIter->getPOCA(Npar).x();
	      float y0 = trackIter->getPOCA(Npar).y();

	      GlobalPoint smearedVertex(x0, y0, smeared_z);
	      smearedTrack.setPOCA( smearedVertex, Npar );

	   }

	   if ( degradeMomentum ) {

		float Pt = trackIter->getMomentum().perp();
		float pz = trackIter->getMomentum().z();
		float Phi = trackIter->getMomentum().phi();

		//float NewPt = Pt;	// the NewPt would be smeared
			/* typically something like :
			   depending on Pt and eta, pick up the resolution (sigma) from Louise's histogram
			   float deltaPt = ran.Gaus(0., sigma);
			   float NewPt = Pt + NsigmaPT * deltaPt :
			*/
		float sigma = 0;

		if(Pt >= 2 && Pt < 5)
		{
		    if(Eta < 1) sigma = 0.00227539 * Eta + 0.0070669;
		    else if(Eta >= 1 && Eta < 1.75) sigma = 0.0111468 * Eta - 0.00141026;
		    else if(Eta >= 1.75 && Eta < 2.2) sigma = 0.00932418 * Eta + 0.0034399;
		    else if(Eta >= 2.2 && Eta < 2.5) sigma = 0.00817958 * Eta + 0.0127676;
		}
		else if(Pt >= 5 && Pt < 10)
		{
		    if(Eta < 1) sigma = 0.00397676 * Eta + 0.00574204;
		    else if(Eta >= 1 && Eta < 1.6) sigma = 0.00503657 * Eta + 0.00507166;
		    else if(Eta >= 1.6 && Eta < 2.1) sigma = 0.0133589 * Eta - 0.00680838;
		    else if(Eta >= 2.1 && Eta < 2.4) sigma = 0.0352536 * Eta - 0.0480062;
		    else if(Eta >= 2.4 && Eta < 2.5) sigma = 0.0332331;
		}
		else if(Pt >= 10 && Pt < 100)
		{
		    if(Eta < 1) sigma = 0.00376492 * Eta + 0.00920416;
		    else if(Eta >= 1 && Eta < 1.5) sigma = 0.00566646 * Eta + 0.00861003;
		    else if(Eta >= 1.65 && Eta < 2) sigma = 0.0406278 * Eta - 0.0509663;
		    else if(Eta >= 2 && Eta < 2.5) sigma = 0.110813 * Eta - 0.194005;
		}
		float deltaPt = ran.Gaus(0., sigma);
		float NewPt = Pt + NsigmaPT * deltaPt * Pt;
		//std::cout << "deltaPt: " << deltaPt << std::endl << "NsigmaPT: " << NsigmaPT << std::endl << "Pt: " << Pt << std::endl << "sigma: " << sigma << std::endl  << NewPt << std::endl;
		float NewPz = pz * NewPt / Pt;

	  	GlobalVector smearedMomentum(GlobalVector::Cylindrical( NewPt,
					Phi, NewPz ));
		smearedTrack.setMomentum( smearedMomentum, Npar );
	   }

	   L1TkTracksForOutput->push_back( smearedTrack );

	}

  iEvent.put( L1TkTracksForOutput, "Level1TTTracks");

}


void L1TrackDegrader::endRun(edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop 

}

// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackDegrader);

#endif







