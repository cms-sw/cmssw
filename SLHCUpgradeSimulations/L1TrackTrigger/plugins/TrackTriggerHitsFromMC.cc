// -*- C++ -*-
//
// Package:    TrackTriggerHitsFromMC
// Class:      TrackTriggerHitsFromMC
// 
/**\class TrackTriggerHitsFromMC TrackTriggerHitsFromMC.cc L1TriggerOffline/L1Trigger/src/TrackTriggerHitsFromMC.cc

 Description: Produce TrackTriggerHit collection from SimTracks

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu Sep 27 16:42:09 CEST 2007
// $Id: TrackTriggerHitsFromMC.cc,v 1.2 2010/02/03 09:46:37 arose Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"
#include "SimDataFormats/SLHC/interface/TrackTriggerCollections.h"

// event setup
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerNaiveGeometry.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerNaiveGeometryRcd.h"

// PDT
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// input data
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/SimpleVector.h"


using namespace std;
using namespace edm;
using namespace HepMC;
//using namespace CLHEP;

//
// class decleration
//

class TrackTriggerHitsFromMC : public edm::EDProducer {
public:
  explicit TrackTriggerHitsFromMC(const edm::ParameterSet&);
  ~TrackTriggerHitsFromMC();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void processGenEvent(const GenEvent* mcEvt);

  bool propagateTrackToLayer(FourVector p, FourVector v, double q, double r, double l);
  
  // ----------member data ---------------------------
  edm::InputTag inputTag_;

  // do pile-up or not
  bool doPU_;

  // mag field strength
  double magField_;

  // tracker parameterisation from ES
  const TrackTriggerNaiveGeometry* geom_;
  edm::ESWatcher<TrackTriggerNaiveGeometryRcd> geomWatcher_;

  // PDT table
  const ParticleDataTable* pdt_;

  // temp hit collection
  TrackTriggerHitCollection* trackHits_;

  // temp values of hit posn
  double hitZ_;
  double hitPhi_;

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
TrackTriggerHitsFromMC::TrackTriggerHitsFromMC(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<TrackTriggerHitCollection>();
   
   inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");

   doPU_ = iConfig.getParameter<bool>("doPileUp");

   // mag field in Tesla
   magField_ = iConfig.getParameter< double >("magField");

   if (doPU_) {
     edm::LogInfo("L1Tracks") << "Going to do pile-up" << std::endl;
   }

}


TrackTriggerHitsFromMC::~TrackTriggerHitsFromMC()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackTriggerHitsFromMC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  LogDebug("L1Tracks") << "Event " << iEvent.id().event() << std::endl;

  // get the tracker parameterisation
  if (geomWatcher_.check(iSetup)) {
    edm::ESHandle<TrackTriggerNaiveGeometry> geomHandle;
    iSetup.get<TrackTriggerNaiveGeometryRcd>().get( geomHandle );
    geom_ = geomHandle.product();
    LogDebug("L1Tracks") << (*geom_) << std::endl;
  }

  LogDebug("L1Tracks") << "ES parameters " << geom_->barrelLayerRadius(0) << std::endl;

  // get the particle data table
  ESHandle <ParticleDataTable> pdt;
  iSetup.getData( pdt );
  pdt_ = pdt.product();
  
  // hit collection
  trackHits_ = new std::vector<TrackTriggerHit>();
  
  if (doPU_) {  // get MC events from CrossingFrame
    edm::Handle<CrossingFrame<edm::HepMCProduct> > cfHepMC;
    iEvent.getByLabel(inputTag_,cfHepMC);
    std::auto_ptr<MixCollection<edm::HepMCProduct> > mmCol(new MixCollection<edm::HepMCProduct>(cfHepMC.product()));
    MixCollection<edm::HepMCProduct>::iterator mmEvt;
    
    // loop over events from mixing module
    int nPU = 0;
    for (mmEvt=mmCol->begin(); mmEvt!=mmCol->end(); mmEvt++) {
      // propagate tracks only for in-time PU
      if (mmEvt.bunch() == 0) {
	processGenEvent(mmEvt->GetEvent());
	nPU++;
      }
    }

    // print number of PU events
    LogDebug("L1Tracks") << "Number of in-time PU events : " << nPU << std::endl;
  }
  else {  // or just get signal HepMCProduct from Event
    Handle< HepMCProduct > evtHandle ;
    iEvent.getByLabel(inputTag_, evtHandle ) ;
    const GenEvent* mcEvt = evtHandle->GetEvent() ;
    processGenEvent(mcEvt);
  }

  // put hits into the Event
  std::auto_ptr<TrackTriggerHitCollection> hits(trackHits_);
  iEvent.put(hits);
  
}

void
TrackTriggerHitsFromMC::processGenEvent(const GenEvent* mcEvt) {

  int nLayers = geom_->nLayers();

  // loop over MC particles
  for ( GenEvent::particle_const_iterator p = mcEvt->particles_begin();
	p != mcEvt->particles_end(); ++p ) {
    
    bool stable = ((*p)->status() == 1 );
    FourVector momentum = (*p)->momentum();
    FourVector vertex;
    if ( (*p)->production_vertex() != 0) {
      vertex = (*p)->production_vertex()->position();
    }
    
    const ParticleData * pData = pdt_->particle( (*p)->pdg_id() );
    double charge = pData->charge();
    
    // only consider stable charged particles
    if (stable && charge!=0) {
      
      LogDebug("L1Tracks") << "MC particle : ID=" << (*p)->pdg_id() << ", q=" << charge << std::endl;
      // loop over layers
      for (int l=0; l<nLayers; ++l) {

	double r = geom_->barrelLayerRadius(l);
	double len = geom_->barrelLayerLength(l);

	// calculate hits - layer 0
	bool hit = propagateTrackToLayer(momentum, 
					 vertex, 
					 charge,
					 r, 
					 len);
	if (hit) {
	  //double theta = ( (r==0. && hitZ_==0.) ? 0. : atan2( r, hitZ_ ) );
	  //double eta = -1 * log( tan( theta / 2 ));

	  /// position
	  LogDebug("L1Tracks") << "Found Hit : layer=" << l << " phi=" << hitPhi_ << " z=" << hitZ_ << std::endl;
	  unsigned id = geom_->barrelTowerId(l, hitZ_, hitPhi_);
	  unsigned row = geom_->barrelPixelRow(id, hitZ_);
	  unsigned col = geom_->barrelPixelColumn(id, hitPhi_);

	  LogDebug("L1Tracks") << "Found Hit : id_=" << id << " row=" << row << " col=" << col << std::endl;	 
	  // make hit
	  //trackHits_->push_back(TrackTriggerHit(id, row, col));
	  trackHits_->push_back(TrackTriggerHit(row, col)); // The new form of TrackTriggerHit has no ID.
	}
	
      }
      
    }
    
  }
  
}


bool TrackTriggerHitsFromMC::propagateTrackToLayer(FourVector p, FourVector v, double q, double r, double l) {

  // check track is charged!
  if (q==0.) return false;

  // did track hit layer
  bool hit = false;

  // track parameters in SI units
  double pz = p.z() * 5.349E-19;
  double pt = p.perp() * 5.349E-19;
  double px = p.x() * 5.349E-19;
  double py = p.y() * 5.349E-19;
  double R = fabs(pt / (q * 1.602E-19 * magField_)) * 100; // calculate directly in SI units

  LogDebug("L1Tracks") << "Track params : px=" << px << ", py=" << py << " pz=" << pz << ", R=" << R << ", q=" << q << std::endl;
  //  std::cout << "Track params : px=" << px << ", py=" << py << " pz=" << pz << ", R=" << R << ", q=" << q << std::endl;

  // vertex coordinates - not in SI units yet!
  double z0 = 0.; //v.z();
  double x0 = 0.; //v.x();
  double y0 = 0.; //v.y();

  // will track reach this layer? If not, return default hit
  if (2*R >= r) {

    // z position of hit
    double hitZ = z0 + r * (pz / pt);
    
    if (fabs(hitZ) < fabs(l/2)) { // hit is within barrel

      // calculate centre of track circle = (X,Y)
      int sgnq = (q > 0 ? 1 : -1);
      double X = ( -1 * sgnq * R * py / sqrt(px*px + py*py) ) + x0;
      double Y = ( sgnq * R * px / sqrt(px*px + py*py)) + y0;      
      double d2 = X*X + Y*Y;
      double w = sqrt(((R+r)*(R+r)-d2)*(d2-(R-r)*(R-r)));

      LogDebug("L1Tracks") << "Track R=" << R << " X=" << X << " Y=" << Y << " w=" << w << std::endl;
      //      std::cout << "Track R=" << R << " X=" << X << " Y=" << Y << " w=" << w << std::endl;

      // intersection of track with layer = (hitX,hitY)
      // http://www.sonoma.edu/users/w/wilsonst/Papers/Geometry/circles/default.html
      // need to check sign of last part of each expression!!!
      double hitX = (X/2) + X*(r-R)/(2*d2) + sgnq*Y*w/(2*d2);
      double hitY = (Y/2) + Y*(r-R)/(2*d2) - sgnq*X*w/(2*d2);
      
      LogDebug("L1Tracks") << "Hit hitX=" << hitX << " hitY=" << hitY << std::endl;
      //      std::cout << "Hit hitX=" << hitX << " hitY=" << hitY << std::endl;

      hit = true;
      hitZ_ = hitZ;
      hitPhi_ = atan2( hitY, hitX );
      
    }
    else { // hit was in endcap - ignore these for now
      hitZ_ = 0.;
      hitPhi_ = 0.;
    }
  }
 
  return hit;

}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackTriggerHitsFromMC::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackTriggerHitsFromMC::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackTriggerHitsFromMC);

