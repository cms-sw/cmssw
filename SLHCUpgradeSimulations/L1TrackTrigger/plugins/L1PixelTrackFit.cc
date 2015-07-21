//////////////////////////
//  Producer by Anders  //
//    July 2014 @ CU    //
//////////////////////////


#ifndef L1PIXELTRACKFIT
#define L1PIXELTRACKFIT

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
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTPixelTrack.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
////////////////////////
// FAST SIMULATION STUFF
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

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
//
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPtConsistency.h"

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
// using namespace std;
// using namespace reco;
using namespace edm;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1PixelTrackFit : public edm::EDProducer
{
public:

  typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
  typedef std::vector< L1TkTrackType >    L1TkTrackCollectionType;


  /// Constructor/destructor
  explicit L1PixelTrackFit(const edm::ParameterSet& iConfig);
  virtual ~L1PixelTrackFit();

protected:
                     
private:

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  void multifit(double invr, double phi0, double d0, double t, double z0,
		std::vector<GlobalPoint> hitL1,
		std::vector<GlobalPoint> hitL2,
		std::vector<GlobalPoint> hitL3,
		std::vector<GlobalPoint> hitL4,
		std::vector<GlobalPoint> hitD1,
		std::vector<GlobalPoint> hitD2,
		std::vector<GlobalPoint> hitD3,
		bool& success,
		double& invrfit,
		double& phi0fit,
		double& d0fit,
		double& tfit,
		double& z0fit,
		double& chisqfit,
		int& nhit,
		double& sigmainvr,
		double& sigmaphi0,
		double& sigmad0,
		double& sigmat,
		double& sigmaz0	
		);
    
    /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

  void invert(double M[5][10],unsigned int n);
  void calculateDerivatives(double rinv, double phi0, double t, double z0,
			    std::vector<GlobalPoint> fithits,
			    std::vector<bool> fitbarrel,
			    double D[5][8], double MinvDt[5][8]);
  void linearTrackFit(double rinv, double phi0, double d0, double t, double z0,
		      double& rinvfit, double& phi0fit, double& d0fit,
		      double& tfit, double& z0fit, double& chisqfit,
		      double& sigmarinv, double& sigmaphi0, double& sigmad0,
		      double& sigmat, double& sigmaz0, 
		      std::vector<GlobalPoint> fithits,
		      std::vector<bool> fitbarrel,
		      double D[5][8], double MinvDt[5][8]);

  void trackfit(double rinv, double phi0, double d0, double t, double z0,
		double& rinvfit, double& phi0fit, double& d0fit,
		double& tfit, double& z0fit, double& chisqfit,
		double& sigmarinv, double& sigmaphi0, double& sigmad0,
		double& sigmat, double& sigmaz0, 
		std::vector<GlobalPoint> fithits,
		std::vector<bool> fitbarrel);

};


//////////////
// CONSTRUCTOR
L1PixelTrackFit::L1PixelTrackFit(edm::ParameterSet const& iConfig) // :   config(iConfig)
{

  produces< std::vector< TTPixelTrack > >( "Level1PixelTracks" ).setBranchAlias("Level1PixelTracks");

}

/////////////
// DESTRUCTOR
L1PixelTrackFit::~L1PixelTrackFit()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void L1PixelTrackFit::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop 

}

////////////
// BEGIN JOB
void L1PixelTrackFit::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
{

}

//////////
// PRODUCE
void L1PixelTrackFit::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  //std::cout << "Entering L1PixelTrackFit::produce"<<std::endl;

  /// Prepare output
  std::auto_ptr< std::vector< TTPixelTrack > > L1PixelTracksForOutput( new std::vector< TTPixelTrack > );


  std::vector<GlobalPoint> cl_pos;
  std::vector<double> cl_phi;
  std::vector<int> cl_type;

  //////////////////////////////////////////////////////////
  // Geometry
  //////////////////////////////////////////////////////////
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);
  ////////////////////////
  // GET MAGNETIC FIELD //
  ////////////////////////
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  //////////////////////////////////////////////////////////
  // RecHits
  //////////////////////////////////////////////////////////
  edm::Handle<SiPixelRecHitCollection> recHits;
  iEvent.getByLabel( "siPixelRecHits", recHits );
  SiPixelRecHitCollection::const_iterator detUnitIt = recHits->begin();
  SiPixelRecHitCollection::const_iterator detUnitItEnd = recHits->end();
  for ( ; detUnitIt != detUnitItEnd; detUnitIt++ ) {
      DetId detId = DetId(detUnitIt->detId());
      SiPixelRecHitCollection::DetSet::const_iterator recHitIt = detUnitIt->begin();
      SiPixelRecHitCollection::DetSet::const_iterator recHitItEnd = detUnitIt->end();
      for ( ; recHitIt != recHitItEnd; ++recHitIt) {
          LocalPoint lp = recHitIt->localPosition();
          GlobalPoint gp = ( (geom.product())->idToDet(detId) )->surface().toGlobal(lp);
	  if ( gp.perp() < 20.0 && fabs(gp.z()) < 55.0){ // reject outer tracker
            cl_pos.push_back(gp);
            cl_phi.push_back(gp.phi());
	    //std::cout << "r z : "<<gp.perp()<<" "<<gp.z()<<std::endl;
	    int type=4;
	    if (gp.perp()<12.0) type=3;
	    if (gp.perp()<8.0) type=2;
	    if (gp.perp()<5.0) type=1;
	    if (fabs(gp.z())>28.0) type=-1;
	    if (fabs(gp.z())>35.0) type=-2;
	    if (fabs(gp.z())>45.0) type=-3;
	    //std::cout << "r z type : "<<gp.perp()<<" "<<gp.z()<<" "<<type<<std::endl;
	    cl_type.push_back(type);
          }
      } // close recHits loop
  } // close detUnits loop


  ///////////////////////////////////////////////////////////
  // L1 tracks 
  //////////////////////////////////////////////////////////
  edm::Handle<L1TkTrackCollectionType> L1TrackHandle;
  //iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);

  //edm::Handle<L1TkTrack_PixelDigi_Collection> L1TrackHandle;
  iEvent.getByLabel("TTTracksFromPixelDigis", "Level1TTTracks", L1TrackHandle);

  //Very inefficient double nested loop...
  L1TkTrackCollectionType::const_iterator iterL1Track;
  int itrack=0;
  for (iterL1Track = L1TrackHandle->begin(); iterL1Track != L1TrackHandle->end(); ++iterL1Track) {
    int npar=5;
    double invr=iterL1Track->getRInv(npar);
    //int charge=1;
    //if (invr<0.0) charge=-1;
    double phi0=iterL1Track->getMomentum(npar).phi();
    double z0=iterL1Track->getPOCA(npar).z();
    double eta=iterL1Track->getMomentum(npar).eta();

    double d0=-iterL1Track->getPOCA(npar).x()*sin(phi0) + 
      iterL1Track->getPOCA(npar).y()*cos(phi0);

 
    double t=sinh(eta);



    std::vector<GlobalPoint> hitL1;
    std::vector<GlobalPoint> hitL2;
    std::vector<GlobalPoint> hitL3;
    std::vector<GlobalPoint> hitL4;
    std::vector<GlobalPoint> hitD1;
    std::vector<GlobalPoint> hitD2;
    std::vector<GlobalPoint> hitD3;

    static const double m_pi=4.0*atan(1.0);

    for(unsigned int i=0;i<cl_pos.size();i++){
      if (cl_type[i]>0) {
	//handle barrel hit here
	double adphi=fabs(cl_phi[i]-phi0);
	if (adphi>0.06&&adphi<6.22) continue;
	double rhit=cl_pos[i].perp();
	if (fabs(z0+t*rhit-cl_pos[i].z())>1.0) continue; //loose cut
	double tmp=asin(0.5*rhit*invr);
	double phi_proj=phi0-tmp+d0/rhit;
	double z_proj=z0+2.0*t*tmp/invr;
	//std::cout<<" rhit zhit z_proj : "<<rhit<<" "<<cl_pos[i].z()<<" "<<z_proj<<std::endl;
	if (fabs(z_proj-cl_pos[i].z())>0.5) continue; // 0.5 cm cut in z!!!
        double dphi=phi_proj-cl_pos[i].phi();
	if (dphi>m_pi) dphi-=2.0*m_pi;
	if (dphi<-m_pi) dphi+=2.0*m_pi;
	if (fabs(dphi*rhit)>0.3) continue; // 3 mm cut in phi!!!
	//found matching hit
	//std::cout << "Barrel Pixel hit: r="
	//  << rhit
	//  << " dz="<<(z_proj-cl_pos[i].z())*10000.0<<" um"
	//  << " rdphi="<<dphi*rhit*10000 <<" um"<<std::endl;
	if (cl_type[i]==1) hitL1.push_back(cl_pos[i]);
	if (cl_type[i]==2) hitL2.push_back(cl_pos[i]);
	if (cl_type[i]==3) hitL3.push_back(cl_pos[i]);
	if (cl_type[i]==4) hitL4.push_back(cl_pos[i]);

      } else {
	//handle disk hit here
        if (fabs(eta)<0.5) continue;
	double adphi=fabs(cl_phi[i]-phi0);
	if (adphi>0.06&&adphi<6.22) continue;
	double zhit=cl_pos[i].z();
	double rhit=cl_pos[i].perp();
	if (fabs((zhit-z0)/t-rhit)>1.0) continue;
	double tmp=0.5*(zhit-z0)*invr/t;
	double r_proj=2.0*sin(tmp)/invr;
	double phi_proj=phi0-tmp+d0/rhit;
	if (fabs(r_proj-rhit)>0.5) continue; // 5 mm cut in r!!!
        double dphi=phi_proj-cl_pos[i].phi();
	if (dphi>m_pi) dphi-=2.0*m_pi;
	if (dphi<-m_pi) dphi+=2.0*m_pi;
	if (fabs(dphi*rhit)>0.3) continue; // 3 mm cut in phi!!!
	//found matching hit
	//std::cout << "Disk Pixel hit: zhit = "<<zhit
	//          << " rhit = "<<rhit
	//          << " phihit = "<<cl_pos[i].phi()
	//	  << " r_proj = "<<r_proj
	//	  << " phi_proj = "<<phi_proj<<std::endl;
	//  << " dz="<<(r_proj-rhit)*10000<<" um"
	//  << " rdphi="<<dphi*rhit*10000 <<" um"<<std::endl;
	if (cl_type[i]==-1) hitD1.push_back(cl_pos[i]);
	if (cl_type[i]==-2) hitD2.push_back(cl_pos[i]);
	if (cl_type[i]==-3) hitD3.push_back(cl_pos[i]);
      }
    }

    bool success=false;
    double invrfit,phi0fit,d0fit,tfit,z0fit,chisqfit;
    double sigmainvr,sigmaphi0,sigmad0,sigmat,sigmaz0;
    int nhit;

    if (fabs(d0)<10.0) {
      multifit(invr,phi0,d0,t,z0,hitL1,hitL2,hitL3,hitL4,
	       hitD1,hitD2,hitD3,success,invrfit,phi0fit,d0fit,tfit,z0fit,
	       chisqfit,nhit,sigmainvr,sigmaphi0,sigmad0,sigmat,sigmaz0);
    }    

    if (success) {

      TTPixelTrack aTrack;

      const GlobalVector aMomentum;

      GlobalPoint thePOCA(-d0fit*sin(phi0fit),d0fit*cos(phi0fit),z0fit);
 
      double pt=0.299792*mMagneticFieldStrength/(100*fabs(invrfit));

      //double ptold=iterL1Track->getMomentum(npar).perp();
      //std::cout << "pt ptold chisq nhit d0old d0 d0fit: "<<pt<<" "<<ptold<<" "
      //	<< chisqfit<<" "<<nhit<<" "
      //	<< d0old*10000<<" "<<d0*10000<<" "<<d0fit*10000<<std::endl;

      GlobalVector theMomentum(GlobalVector::Cylindrical(pt,phi0fit,pt*tfit));

      edm::Ref< L1TkTrackCollectionType > L1TrackPtr( L1TrackHandle, itrack) ;      
      //std::cout << "chisqfit : " << chisqfit << std::endl;

      aTrack.init(L1TrackPtr,theMomentum,thePOCA,invrfit,chisqfit,nhit,
		  sigmainvr,sigmaphi0,sigmad0,sigmat,sigmaz0);

      L1PixelTracksForOutput->push_back(aTrack);

    }

    itrack++;
    
  }// end loop L1 tracks





  iEvent.put( L1PixelTracksForOutput, "Level1PixelTracks");

  //std::cout << "Exiting L1PixelTrackFit::produce"<<std::endl;


} /// End of produce()


void L1PixelTrackFit::multifit(double rinv, double phi0, double d0, 
			       double t, double z0,
			       std::vector<GlobalPoint> hitL1,
			       std::vector<GlobalPoint> hitL2,
			       std::vector<GlobalPoint> hitL3,
			       std::vector<GlobalPoint> hitL4,
			       std::vector<GlobalPoint> hitD1,
			       std::vector<GlobalPoint> hitD2,
			       std::vector<GlobalPoint> hitD3,		
			       bool& success,
			       double& invrfinal,
			       double& phi0final,
			       double& d0final,
			       double& tfinal,
			       double& z0final,
			       double& chisqfinal,
			       int& nhit,
			       double& sigmainvr,
			       double& sigmaphi0,
			       double& sigmad0,
			       double& sigmat,
			       double& sigmaz0
			       ) {
  
  success=false;

  /*
  static ofstream out("fitdump.txt");

  out << "L1Track: " << rinv << " " << phi0 << " " << t << " " << z0 << std::endl;
  out << hitL1.size() << std::endl;
  for(unsigned int i=0;i<hitL1.size();i++) {
    out << hitL1[i].x() << " " << hitL1[i].y() << " " << hitL1[i].z() << std::endl;
  }
  out << hitL2.size() << std::endl;
  for(unsigned int i=0;i<hitL2.size();i++) {
    out << hitL2[i].x() << " " << hitL2[i].y() << " " << hitL2[i].z() << std::endl;
  }
  out << hitL3.size() << std::endl;
  for(unsigned int i=0;i<hitL3.size();i++) {
    out << hitL3[i].x() << " " << hitL3[i].y() << " " << hitL3[i].z() << std::endl;
  }
  out << hitL4.size() << std::endl;
  for(unsigned int i=0;i<hitL4.size();i++) {
    out << hitL4[i].x() << " " << hitL4[i].y() << " " << hitL4[i].z() << std::endl;
  }
  out << hitD1.size() << std::endl;
  for(unsigned int i=0;i<hitD1.size();i++) {
    out << hitD1[i].x() << " " << hitD1[i].y() << " " << hitD1[i].z() << std::endl;
  }
  out << hitD2.size() << std::endl;
  for(unsigned int i=0;i<hitD2.size();i++) {
    out << hitD2[i].x() << " " << hitD2[i].y() << " " << hitD2[i].z() << std::endl;
  }
  out << hitD3.size() << std::endl;
  for(unsigned int i=0;i<hitD3.size();i++) {
    out << hitD3[i].x() << " " << hitD3[i].y() << " " << hitD3[i].z() << std::endl;
  }

  */

  std::vector<GlobalPoint> hits[7];
  bool barrel[7];
  
  barrel[0]=true;
  barrel[1]=true;
  barrel[2]=true;
  barrel[3]=true;
  barrel[4]=false;
  barrel[5]=false;
  barrel[6]=false;

  hits[0]=hitL1;
  hits[1]=hitL2;
  hits[2]=hitL3;
  hits[3]=hitL4;
  hits[4]=hitD1;
  hits[5]=hitD2;
  hits[6]=hitD3;
  
  //for (int jj=0;jj<7;jj++) {
  //  std::cout  << "hits["<<jj<<"].size() = "<<hits[jj].size()<<std::endl;
  //}
 
  //sort on number of hits per layer

  bool more=false;
  do {
    more=false;
    for(int i=0;i<6;i++) {
      if (hits[i].size()<hits[i+1].size()) {
	more=true;
	std::vector<GlobalPoint> tmp=hits[i];
	hits[i]=hits[i+1];
	hits[i+1]=tmp;
	bool tmpb=barrel[i];
	barrel[i]=barrel[i+1];
	barrel[i+1]=tmpb;
      }
    }
  } while(more);

  //now start fitting

  double bestChisqdof=1e30;
  int i0best;
  int i1best;
  int i2best;
  int i3best;
  int nhitsbest=0;
  double rinvbest=9999.9;
  double phi0best=9999.9;
  double d0best=9999.9;
  double tbest=9999.9;
  double z0best=9999.9;
  double chisqbest=9999.9;

    
  for(unsigned int i0=0;i0<hits[0].size()+1;i0++) {
    for(unsigned int i1=0;i1<hits[1].size()+1;i1++) {
      for(unsigned int i2=0;i2<hits[2].size()+1;i2++) {
	for(unsigned int i3=0;i3<hits[3].size()+1;i3++) {
	  int npixel=0;
	  if (i0!=hits[0].size()) npixel++;
	  if (i1!=hits[1].size()) npixel++;
	  if (i2!=hits[2].size()) npixel++;
	  if (i3!=hits[3].size()) npixel++;
	  //std::cout << "npixel i0 i1 i2 i3 : "<<npixel<<" "<<i0<<" "
	  //	    <<i1<<" "<<i2<<" "<<i3<<std::endl;
	  if (npixel<3) continue;
	  std::vector<GlobalPoint> fithits;
	  std::vector<bool> fitbarrel;
	  if (i0<hits[0].size()) {
	    fithits.push_back(hits[0][i0]);
	    fitbarrel.push_back(barrel[0]);
	  }
	  if (i1<hits[1].size()) {
	    fithits.push_back(hits[1][i1]);
	    fitbarrel.push_back(barrel[1]);
	  }
	  if (i2<hits[2].size()) {
	    fithits.push_back(hits[2][i2]);
	    fitbarrel.push_back(barrel[2]);
	  }
	  if (i3<hits[3].size()) {
	    fithits.push_back(hits[3][i3]);
	    fitbarrel.push_back(barrel[3]);
	  }
	  double rinvfit,phi0fit,d0fit,tfit,z0fit,chisqfit;
	  trackfit(rinv,phi0,d0,t,z0,
		   rinvfit,phi0fit,d0fit,tfit,z0fit,chisqfit,
		   sigmainvr,sigmaphi0,sigmad0,sigmat,sigmaz0,
		   fithits,fitbarrel);
	  double chisqdof=chisqfit/(2.0*npixel-5.0); //this a bit arbitrary
	  if (chisqdof<bestChisqdof) {
	    bestChisqdof=chisqdof;
	    nhitsbest=npixel;
	    i0best=i0;
	    i1best=i1;
	    i2best=i2;
	    i3best=i3;
	    rinvbest=rinvfit;
	    phi0best=phi0fit;
	    d0best=d0fit;
	    tbest=tfit;
	    z0best=z0fit;
	    chisqbest=chisqfit;
	  }
	  //std::cout << "Original: rinv="<<rinv
	  //	    <<" phi0="<<phi0
	  //    <<" d0="<<d0*10000.0
	  //    <<" um t="<<t
	  //    <<" z0="<<z0 << std::endl;
	  //std::cout << "PixelFit: rinv="<<rinvfit
	  //    <<" phi0="<<phi0fit
	  //    <<" d0="<<d0fit*10000.0
	  //    <<" um t="<<tfit
	  //    <<" z0="<<z0fit 
	  //    <<" chisq="<<chisqfit<<std::endl;
	}
      }
    }
  }

  if (bestChisqdof<1e29) {
    
    if (0) {
      std::cout << i0best<<i1best<<i2best<<i3best<<std::endl;
    }

    success=true;

    invrfinal=rinvbest;
    phi0final=phi0best;
    d0final=d0best;
    tfinal=tbest;
    z0final=z0best;
    chisqfinal=chisqbest;
    nhit=nhitsbest;

    //std::cout << "Found best fit:"<<std::endl;
    //std::cout << "Original: rinv="<<rinv
    //      <<" phi0="<<phi0
    //      <<" d0="<<d0*10000.0
    //      <<" um t="<<t
    //      <<" z0="<<z0 << std::endl;
    //std::cout << "PixelFit: chisqdof="<<bestChisqdof
    //      <<" rinv="<<rinvbest
    //      <<" phi0="<<phi0best
    //      <<" d0="<<d0best*10000.0
    //      <<" um t="<<tbest
    //      <<" z0="<<z0best << std::endl;
    
    
  }




}


void L1PixelTrackFit::invert(double M[5][10],unsigned int n){
  
  assert(n<=5);

  unsigned int i,j,k;
  double ratio,a;
    
  for(i = 0; i < n; i++){
    for(j = n; j < 2*n; j++){
      if(i==(j-n))
	M[i][j] = 1.0;
      else
	M[i][j] = 0.0;
    }
  }
  
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i!=j){
	ratio = M[j][i]/M[i][i];
	for(k = 0; k < 2*n; k++){
	  M[j][k] -= ratio * M[i][k];
	}
      }
    }
  }

  for(i = 0; i < n; i++){
    a = M[i][i];
    for(j = 0; j < 2*n; j++){
      M[i][j] /= a;
    }
  }
}



void L1PixelTrackFit::calculateDerivatives(double rinv, double phi0, double t, double z0,
					   std::vector<GlobalPoint> fithits,
					   std::vector<bool> fitbarrel,
					   double D[5][8], double MinvDt[5][8]){
  
  double M[5][10];

  unsigned int n=fithits.size();
  
  assert(n<=4);
  
  
  int j=0;

  for(unsigned int i=0;i<n;i++) {

    double ri=fithits[i].perp();
    double zi=fithits[i].z();
    double phii=fithits[i].phi();

    double sigmax=0.007/sqrt(12.0);
    double sigmaz=0.01/sqrt(12.0);
    
    if (fitbarrel[i]){
      //here we handle a barrel hit
      
      //first we have the phi position
      D[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv*rinv)/sigmax;
      D[1][j]=ri/sigmax;
      D[2][j]=0.0;
      D[3][j]=0.0;
      D[4][j]=-1.0/sigmax;
      j++;
      //second the z position
      D[0][j]=0.0;
      D[1][j]=0.0;
      D[2][j]=(2/rinv)*asin(0.5*ri*rinv)/sigmaz;
      D[3][j]=1.0/sigmaz;
      D[4][j]=0.0;
      j++;
    }
    else {
      //here we handle a disk hit
      //first we have the r position
      
      double r_track=2.0*sin(0.5*rinv*(zi-z0)/t)/rinv;
      double phi_track=phi0-0.5*rinv*(zi-z0)/t;
      
      double rmultiplier=sin(phi_track-phii);
      double phimultiplier=r_track*cos(phi_track-phii);
      
      double drdrinv=-2.0*sin(0.5*rinv*(zi-z0)/t)/(rinv*rinv)
	+(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(rinv*t);
      double drdphi0=0;
      double drdt=-(zi-z0)*cos(0.5*rinv*(zi-z0)/t)/(t*t);
      double drdz0=-cos(0.5*rinv*(zi-z0)/t)/t;
      
      double dphidrinv=-0.5*(zi-z0)/t;
      double dphidphi0=1.0;
      double dphidt=0.5*rinv*(zi-z0)/(t*t);
      double dphidz0=0.5*rinv/t;
	
      D[0][j]=drdrinv/sigmaz;
      D[1][j]=drdphi0/sigmaz;
      D[2][j]=drdt/sigmaz;
      D[3][j]=drdz0/sigmaz;
      D[4][j]=0;
      j++;
      //second the rphi position
      D[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigmax;
      D[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigmax;
      D[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigmax;
      D[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigmax;
      D[4][j]=-1.0/sigmax;
      j++;
    }

    //cout << "Exact rinv derivative: "<<i<<" "<<D[0][j-2]<<" "<<D[0][j-1]<<endl;
    //cout << "Exact phi0 derivative: "<<i<<" "<<D[1][j-2]<<" "<<D[1][j-1]<<endl;
    //cout << "Exact t derivative   : "<<i<<" "<<D[2][j-2]<<" "<<D[2][j-1]<<endl;
    //cout << "Exact z0 derivative  : "<<i<<" "<<D[3][j-2]<<" "<<D[3][j-1]<<endl;
	
	
  }
    
  unsigned int npar=5;

  for(unsigned int i1=0;i1<npar;i1++){
    for(unsigned int i2=0;i2<npar;i2++){
      M[i1][i2]=0.0;
      for(unsigned int j=0;j<2*n;j++){
	M[i1][i2]+=D[i1][j]*D[i2][j];	  
      }
    }
  }
  
  //Approximate errors from L1 tracks
  M[0][0]+=1.0/pow(0.03*rinv,2);
  M[1][1]+=1.0/pow(0.0005,2);
  M[2][2]+=1.0/pow(0.0025,2);

  invert(M,npar);
  
  for(unsigned int j=0;j<2*n;j++) {
    for(unsigned int i1=0;i1<npar;i1++) {
      MinvDt[i1][j]=0.0;
      for(unsigned int i2=0;i2<npar;i2++) {
	MinvDt[i1][j]+=M[i1][i2+npar]*D[i2][j];
      }
    }
  }
  
}


void L1PixelTrackFit::linearTrackFit(double rinv, double phi0, double d0,
				     double t, double z0,
				     double& rinvfit, double& phi0fit, double& d0fit,
				     double& tfit, double& z0fit, double& chisqfit,
				     double& sigmarinv, double& sigmaphi0, double& sigmad0,
				     double& sigmat, double& sigmaz0, 
				     std::vector<GlobalPoint> fithits,
				     std::vector<bool> fitbarrel,
				     double D[5][8], double MinvDt[5][8]){
  

  unsigned int n=fithits.size();
  
  //Next calculate the residuals
  
  double delta[40];
  
  double chisq=0;
  
  //int charge=1;
  //if (rinv<0.0) charge=-1;
  
  unsigned int j=0;

  //double vm[8];

  for(unsigned int i=0;i<n;i++) {
    double ri=fithits[i].perp();
    double zi=fithits[i].z();
    double phii=fithits[i].phi();
    //std::cout << "phii= "<<phii<<" x="<<fithits[i].x()
    //	      <<" y="<<fithits[i].y()<<std::endl;
    double sigmax=0.007/sqrt(12.0);
    double sigmaz=0.01/sqrt(12.0);
    
    
    if (fitbarrel[i]) {
      //we are dealing with a barrel stub

      static const double two_pi=8.0*atan(1.0);

      double deltaphi=phi0-asin(0.5*ri*rinv)+d0/ri-phii;
      if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
      if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
      //std::cout << "phi0="<<phi0<<" phii="<<phii<<std::endl;
      assert(fabs(deltaphi)<0.1*two_pi);
      
      //vm[j]=sigmax*sigmax;
      delta[j++]=-ri*deltaphi/sigmax;
      //vm[j]=sigmaz*sigmaz;
      delta[j++]=(z0+(2.0/rinv)*t*asin(0.5*ri*rinv)-zi)/sigmaz;
      
      
      //numerical derivative check
      /*
      for (int iii=0;iii<0;iii++){

	double drinv=0.0;
	double dphi0=0.0;
	double dt=0.0;
	double dz0=0.0;

	if (iii==0) drinv=0.001*fabs(rinv_);
	if (iii==1) dphi0=0.001;
	if (iii==2) dt=0.001;
	if (iii==3) dz0=0.01;
	  
	double deltaphi=phi0_+dphi0-asin(0.5*ri*(rinv_+drinv))-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	assert(fabs(deltaphi)<0.1*two_pi);

	double delphi=ri*deltaphi/sigmax;
	double deltaz=(z0_+dz0+(2.0/(rinv_+drinv))*(t_+dt)*asin(0.5*ri*(rinv_+drinv))-zi)/sigmaz;


	if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			 <<(delphi-delta[j-2])/drinv<<" "
			 <<(deltaz-delta[j-1])/drinv<<endl;
	
	if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			 <<(delphi-delta[j-2])/dphi0<<" "
			 <<(deltaz-delta[j-1])/dphi0<<endl;
	
	if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			 <<(delphi-delta[j-2])/dt<<" "
			 <<(deltaz-delta[j-1])/dt<<endl;
	
	if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			 <<(delphi-delta[j-2])/dz0<<" "
			 <<(deltaz-delta[j-1])/dz0<<endl;
	
      }
      */


    }
    else {
      //we are dealing with a disk hit
      
      double r_track=2.0*sin(0.5*rinv*(zi-z0)/t)/rinv;
      double phi_track=phi0-0.5*rinv*(zi-z0)/t+d0/ri;
      
      double Delta=r_track*sin(phi_track-phii);
      
      //vm[j]=sigmaz*sigmaz;
      delta[j++]=(r_track-ri)/sigmaz;
      //vm[j]=sigmax*sigmax;
      delta[j++]=-Delta/sigmax;

      //numerical derivative check

      /*      
      for (int iii=0;iii<0;iii++){
	
	double drinv=0.0;
	double dphi0=0.0;
	double dt=0.0;
	double dz0=0.0;
	
	if (iii==0) drinv=0.001*fabs(rinv_);
	if (iii==1) dphi0=0.001;
	if (iii==2) dt=0.001;
	if (iii==3) dz0=0.01;
	
	r_track=2.0*sin(0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt))/(rinv_+drinv);
	//cout <<"t_track 2: "<<r_track<<endl;
	phi_track=phi0_+dphi0-0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt);
	
	iphi=stubs_[i].iphi();
	
	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
	
	if (stubs_[i].z()>0.0) Deltai=-Deltai;
	theta0=asin(Deltai/ri);
	
	Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			 <<((r_track-ri)/sigmaz-delta[j-2])/drinv<<" "
			 <<(Delta/sigmax-delta[j-1])/drinv<<endl;
	
	if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			 <<((r_track-ri)/sigmaz-delta[j-2])/dphi0<<" "
			 <<(Delta/sigmax-delta[j-1])/dphi0<<endl;
	
	if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			 <<((r_track-ri)/sigmaz-delta[j-2])/dt<<" "
			 <<(Delta/sigmax-delta[j-1])/dt<<endl;
	
	if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			 <<((r_track-ri)/sigmaz-delta[j-2])/dz0<<" "
			 <<(Delta/sigmax-delta[j-1])/dz0<<endl;
	
      }
      */
      
    }
    
    chisq+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]);
    
  }
  
  double drinv=0.0;
  double dphi0=0.0;
  double dd0=0.0;
  double dt=0.0;
  double dz0=0.0;

  double drinv_cov=0.0;
  double dphi0_cov=0.0;
  double dd0_cov=0.0;
  double dt_cov=0.0;
  double dz0_cov=0.0;
    


  for(unsigned int j=0;j<2*n;j++) {
    drinv-=MinvDt[0][j]*delta[j];
    dphi0-=MinvDt[1][j]*delta[j];
    dt-=MinvDt[2][j]*delta[j];
    dz0-=MinvDt[3][j]*delta[j];
    dd0-=MinvDt[4][j]*delta[j];
    
    drinv_cov+=D[0][j]*delta[j];
    dphi0_cov+=D[1][j]*delta[j];
    dt_cov+=D[2][j]*delta[j];
    dz0_cov+=D[3][j]*delta[j];
    dd0_cov+=D[4][j]*delta[j];
  }
    
  double vpar[5];

  for(unsigned ipar=0;ipar<5;ipar++){
    vpar[ipar]=0.0;
    for(unsigned int j=0;j<2*n;j++) {
      //vpar[ipar]+=MinvDt[ipar][j]*vm[j]*MinvDt[ipar][j];
      vpar[ipar]+=MinvDt[ipar][j]*MinvDt[ipar][j];
    }
  }

  sigmarinv=sqrt(vpar[0]);
  sigmaphi0=sqrt(vpar[1]);
  sigmat=sqrt(vpar[2]);
  sigmaz0=sqrt(vpar[3]);
  sigmad0=sqrt(vpar[4]);
  


  //std::cout << "sigma d0, sigma z0 : "
  //	    << sqrt(vpar[4])*10000<<" "
  //	    << sqrt(vpar[3])*10000<<std::endl;
  

  double deltaChisq=drinv*drinv_cov+
    dphi0*dphi0_cov+
    dt*dt_cov+
    dz0*dz0_cov+
    dd0*dd0_cov;
  
  //drinv=0.0; dphi0=0.0; dt=0.0; dz0=0.0;
  
  rinvfit=rinv+drinv;
  phi0fit=phi0+dphi0;
  tfit=t+dt;
  z0fit=z0+dz0;
  d0fit=d0+dd0; 

  //std::cout << "d0fit d0 dd0 : "<<d0fit<<" "<<d0<<" "<<dd0<<std::endl;
   
  chisqfit=(chisq+deltaChisq);
    
 
  //cout << "Trackfit:"<<endl;
  //cout << "rinv_ drinv: "<<rinv_<<" "<<drinv<<endl;
  //cout << "phi0_ dphi0: "<<phi0_<<" "<<dphi0<<endl;
  //cout << "t_ dt      : "<<t_<<" "<<dt<<endl;
  //cout << "z0_ dz0    : "<<z0_<<" "<<dz0<<endl;
  
}

void L1PixelTrackFit::trackfit(double rinv, double phi0, double d0,
			       double t, double z0,
			       double& rinvfit, double& phi0fit, double& d0fit,
			       double& tfit, double& z0fit, double& chisqfit,
			       double& sigmarinv, double& sigmaphi0, double& sigmad0,
			       double& sigmat, double& sigmaz0, 
			       std::vector<GlobalPoint> fithits,
			       std::vector<bool> fitbarrel){

  double D[5][8];
  double MinvDt[5][8];



  calculateDerivatives(rinv, phi0, t, z0,fithits,fitbarrel,D,MinvDt);
  linearTrackFit(rinv, phi0, d0, t, z0,
		 rinvfit, phi0fit, d0fit, tfit, z0fit, chisqfit,
		 sigmarinv, sigmaphi0, sigmad0, sigmat, sigmaz0, 
		 fithits,fitbarrel,D,MinvDt);

}




// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1PixelTrackFit);

#endif


