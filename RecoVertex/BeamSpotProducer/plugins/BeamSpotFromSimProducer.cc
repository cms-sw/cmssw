//////////////////////////
//  Producer by A. Ryd  //
//    Jan 2013 @ CU     //
//////////////////////////


////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include <string>

//////////////
// NAMESPACES
using namespace edm;
using namespace std;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class BeamSpotFromSimProducer : public edm::EDProducer
{
public:

  // point in the space
  typedef math::XYZPoint Point;
  enum { dimension = 7 };
  typedef math::Error<dimension>::type CovarianceMatrix;


  /// Constructor/destructor
  explicit BeamSpotFromSimProducer(const edm::ParameterSet& iConfig);
  virtual ~BeamSpotFromSimProducer();

protected:
                     
private:

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

  double meanX_;
  double meanY_;
  double meanZ_;

  double sigmaX_;
  double sigmaY_;
  double sigmaZ_;

  double dxdz_;
  double dydz_;

  Point point_;
  CovarianceMatrix error_;

};


//////////////
// CONSTRUCTOR
BeamSpotFromSimProducer::BeamSpotFromSimProducer(edm::ParameterSet const& iConfig) // :   config(iConfig)
{
  produces< reco::BeamSpot >( "BeamSpot" ).setBranchAlias("BeamSpot");
}

/////////////
// DESTRUCTOR
BeamSpotFromSimProducer::~BeamSpotFromSimProducer()
{
}  

//////////
// END JOB
void BeamSpotFromSimProducer::endRun(edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop
}

////////////
// BEGIN JOB
void BeamSpotFromSimProducer::beginRun(edm::Run& run, const edm::EventSetup& iSetup )
{
}

//////////
// PRODUCE
void BeamSpotFromSimProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  static bool gotProcessParameterSet=false;

  if(!gotProcessParameterSet){

    //fetch the real process name using the processHistory accessor of 
    //the edm::Event, then find the 	
    // last-1 entry and ask for it's processName

    unsigned int nProcess=iEvent.processHistory().size();

    for (unsigned int iProcess=0;iProcess<nProcess;iProcess++) {

      if (gotProcessParameterSet) continue;

      std::string nameProcess = iEvent.processHistory()[iProcess].processName();
      //  std::cout << "Process "<<i<<" name: "
      //		<<iEvent.processHistory()[i].processName()<<std::endl;

    
      // Now ask the edm::Event for the top level parameter set of this process, 
      //return it in ps

      edm::ParameterSet ps;
      iEvent.getProcessParameterSet(nameProcess,ps);


      if (ps.exists("VtxSmeared")) {

	string type = ps.getParameterSet("VtxSmeared").getParameter<string>("@module_type");

	if (type=="HLLHCEvtVtxGenerator") {

	  double eproton = ps.getParameterSet("VtxSmeared").getParameter<double>("EprotonInGeV");

	  double betastarx = ps.getParameterSet("VtxSmeared").getParameter<double>("BetaStarCrossingPlaneInm");

	  double betastary = ps.getParameterSet("VtxSmeared").getParameter<double>("BetaStarParallelPlaneInm");

	  double epsilonx = ps.getParameterSet("VtxSmeared").getParameter<double>("NormalizedEmittanceCrossingPlaneInum");

	  double epsilony = ps.getParameterSet("VtxSmeared").getParameter<double>("NormalizedEmittanceParallelPlaneInum");

	  double betagamma=eproton/0.94; //FIXME

	  double SigmaX = sqrt(epsilonx*betastarx/betagamma);
	  double SigmaY = sqrt(epsilony*betastary/betagamma);

	  double SigmaZ = ps.getParameterSet("VtxSmeared").getParameter<double>("ZsizeInm");
	  
	  meanX_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanXIncm");
	  meanY_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanYIncm");
	  meanZ_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanZIncm");
	  
	  double HalfCrossingAngle = ps.getParameterSet("VtxSmeared").getParameter<double>("HalfCrossingAngleInurad");
	  double CrabAngle = ps.getParameterSet("VtxSmeared").getParameter<double>("CrabAngleCrossingPlaneInurad");
      
	  static const double sqrtOneHalf=sqrt(0.5);

	  double sinbeta=sin(CrabAngle);
	  double cosbeta=cos(CrabAngle);
	  double cosalpha=cos(HalfCrossingAngle);
	  double sinamb=sin(HalfCrossingAngle-CrabAngle);
	  double cosamb=cos(HalfCrossingAngle-CrabAngle);
	  
	  sigmaX_=sqrtOneHalf*hypot(SigmaZ*sinbeta,SigmaX*cosbeta)/cosalpha;
	  
	  sigmaY_=sqrtOneHalf*SigmaY;

	  sigmaZ_=sqrtOneHalf/hypot(sinamb/SigmaX,cosamb/SigmaZ);
  
	  dxdz_=0.0;
	  dydz_=0.0;
    
	  point_=Point(meanX_,meanY_,meanZ_);

	  for (unsigned int j=0; j<7; j++) {
	    for (unsigned int k=j; k<7; k++) {
	      error_(j,k) = 0.0;
	    }
	  }


	  //arbitrarily set position errors to 1/10 of width.
	  error_(0,0)=0.1*sigmaX_;
	  error_(1,1)=0.1*sigmaY_;
	  error_(2,2)=0.1*sigmaZ_;
    
	  //arbitrarily set width errors to 1/10 of width.
	  error_(3,3)=0.1*sigmaZ_;
	  error_(6,6)=0.1*sigmaX_;
      
	  //arbitrarily set error on beam axis direction to 1/100 of
	  //beam aspect ratio    
	  error_(4,4)=0.01*sigmaX_/sigmaZ_;
	  error_(5,5)=error_(4,4);
	  gotProcessParameterSet=true;
	}
	else if (type=="GaussEvtVtxGenerator") {

	  sigmaX_ = ps.getParameterSet("VtxSmeared").getParameter<double>("SigmaX");
	  sigmaY_ = ps.getParameterSet("VtxSmeared").getParameter<double>("SigmaY");
	  sigmaZ_ = ps.getParameterSet("VtxSmeared").getParameter<double>("SigmaZ");
      
	  meanX_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanX");
	  meanY_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanY");
	  meanZ_ = ps.getParameterSet("VtxSmeared").getParameter<double>("MeanZ");
      
	  dxdz_=0.0;
	  dydz_=0.0;
    
	  point_=Point(meanX_,meanY_,meanZ_);

	  for (unsigned int j=0; j<7; j++) {
	    for (unsigned int k=j; k<7; k++) {
	      error_(j,k) = 0.0;
	    }
	  }


	  //arbitrarily set position errors to 1/10 of width.
	  error_(0,0)=0.1*sigmaX_;
	  error_(1,1)=0.1*sigmaY_;
	  error_(2,2)=0.1*sigmaZ_;
    
	  //arbitrarily set width errors to 1/10 of width.
	  error_(3,3)=0.1*sigmaZ_;
	  error_(6,6)=0.1*sigmaX_;
      
	  //arbitrarily set error on beam axis direction to 1/100 of
	  //beam aspect ratio    
	  error_(4,4)=0.01*sigmaX_/sigmaZ_;
	  error_(5,5)=error_(4,4);
	  gotProcessParameterSet=true;
	}
	else {
	  LogError("BeamSpotFromSimProducer") <<"In BeamSpotFromSimProducer type="<<type
					      <<" don't know what to do!"<<std::endl; 
	}
      }
    }

    if (!gotProcessParameterSet) {
      LogError("BeamSpotFromSimProducer") <<"In BeamSpotFromSimProducer could not find the parameter set 'VtxSmeared'"<<std::endl; 
    }

  }


  /// Prepare output
  std::auto_ptr< reco::BeamSpot > BeamSpotForOutput( new reco::BeamSpot(point_,
									sigmaZ_,
									dxdz_,
									dydz_,
									sigmaX_,
									error_,
									reco::BeamSpot::Fake));


  iEvent.put( BeamSpotForOutput, "BeamSpot");

} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(BeamSpotFromSimProducer);

