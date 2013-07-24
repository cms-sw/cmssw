/*
 * \file EcalTBMCInfoProducer.cc
 *
 * $Id: EcalTBMCInfoProducer.cc,v 1.12 2009/04/03 09:38:15 fabiocos Exp $
 *
*/

#include "SimG4CMS/EcalTestBeam/interface/EcalTBMCInfoProducer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Math/interface/Point3D.h"

using namespace std;
using namespace cms;

EcalTBMCInfoProducer::EcalTBMCInfoProducer(const edm::ParameterSet& ps) : flatDistribution_(0) {
  
  produces<PEcalTBInfo>();

  edm::FileInPath CrystalMapFile = ps.getParameter<edm::FileInPath>("CrystalMapFile");
  GenVtxLabel = ps.getUntrackedParameter<string>("moduleLabelVtx","source");
  double fMinEta = ps.getParameter<double>("MinEta");
  double fMaxEta = ps.getParameter<double>("MaxEta");
  double fMinPhi = ps.getParameter<double>("MinPhi");
  double fMaxPhi = ps.getParameter<double>("MaxPhi");
  beamEta = (fMaxEta+fMinEta)/2.;
  beamPhi = (fMaxPhi+fMinPhi)/2.;
  beamTheta = 2.0*atan(exp(-beamEta));
  beamXoff = ps.getParameter<double>("BeamMeanX");
  beamYoff = ps.getParameter<double>("BeamMeanX");
   
  string fullMapName = CrystalMapFile.fullPath();
  theTestMap = new EcalTBCrystalMap(fullMapName);
  crysNumber = 0;

  double deltaEta = 999.;
  double deltaPhi = 999.;
  for ( int cryIndex = 1; cryIndex <= EcalTBCrystalMap::NCRYSTAL; ++cryIndex) {
    double eta = 0;
    double phi = 0.;
    theTestMap->findCrystalAngles(cryIndex, eta, phi);
    if ( fabs(beamEta - eta) < deltaEta && fabs(beamPhi - phi) < deltaPhi ) {
      deltaEta = fabs(beamEta - eta);
      deltaPhi = fabs(beamPhi - phi);
      crysNumber = cryIndex;
    }
    else if (fabs(beamEta - eta)<deltaEta && fabs(beamPhi - phi)>deltaPhi ) {
      if ( fabs(beamPhi - phi) < 0.017 ) {
        deltaEta = fabs(beamEta - eta);
        deltaPhi = fabs(beamPhi - phi);
        crysNumber = cryIndex;
      }
    }
    else if (fabs(beamEta - eta)>deltaEta && fabs(beamPhi - phi)<deltaPhi ) {
      if ( fabs(beamEta - eta) < 0.017 ) {
        deltaEta = fabs(beamEta - eta);
        deltaPhi = fabs(beamPhi - phi);
        crysNumber = cryIndex;
      }
    }
  }

  edm::LogInfo("EcalTBInfo") << "Initialize TB MC ECAL info producer with parameters: \n"
                             << "Crystal map file:  " << CrystalMapFile << "\n"
                             << "Beam average eta = " << beamEta << "\n"
                             << "Beam average phi = " << beamPhi << "\n"
                             << "Corresponding to crystal number = " << crysNumber << "\n"
                             << "Beam X offset =    " << beamXoff << "\n"
                             << "Beam Y offset =    " << beamYoff;

  // rotation matrix to move from the CMS reference frame to the test beam one

  double xx = -cos(beamTheta)*cos(beamPhi);
  double xy = -cos(beamTheta)*sin(beamPhi);
  double xz = sin(beamTheta);
  
  double yx = sin(beamPhi);
  double yy = -cos(beamPhi);
  double yz = 0.;
  
  double zx = sin(beamTheta)*cos(beamPhi);
  double zy = sin(beamTheta)*sin(beamPhi);
  double zz = cos(beamTheta);

  fromCMStoTB = new ROOT::Math::Rotation3D(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  // random number
  edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "EcalTBMCInfoProducer requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }
   CLHEP::HepRandomEngine& engine = rng->getEngine();
   flatDistribution_ = new CLHEP::RandFlat(engine);

}
 
EcalTBMCInfoProducer::~EcalTBMCInfoProducer() {

  delete flatDistribution_;
  delete theTestMap;
  
}

 void EcalTBMCInfoProducer::produce(edm::Event & event, const edm::EventSetup& eventSetup)
{
  auto_ptr<PEcalTBInfo> product(new PEcalTBInfo());

  // Fill the run information

  product->setCrystal(crysNumber);

  product->setBeamDirection(beamEta, beamPhi);
  product->setBeamOffset(beamXoff, beamYoff);

  // Compute the event x,y vertex coordinates in the beam reference system
  // e.g. in the place orthogonal to the beam average direction

  partXhodo = partYhodo = 0.;

  edm::Handle<edm::HepMCProduct> GenEvt;
  event.getByLabel(GenVtxLabel,GenEvt);

  const HepMC::GenEvent* Evt = GenEvt->GetEvent() ;
  HepMC::GenEvent::vertex_const_iterator Vtx = Evt->vertices_begin();

  math::XYZPoint eventCMSVertex((*Vtx)->position().x(),
                                (*Vtx)->position().y(),
                                (*Vtx)->position().z());
  
  LogDebug("EcalTBInfo") << "Generated vertex position = " 
                         << eventCMSVertex.x() << " " 
                         << eventCMSVertex.y() << " " 
                         << eventCMSVertex.z();

  math::XYZPoint eventTBVertex = (*fromCMStoTB)*eventCMSVertex;

  LogDebug("EcalTBInfo") << "Rotated vertex position   = "
                         << eventTBVertex.x() << " " 
                         << eventTBVertex.y() << " " 
                         << eventTBVertex.z();

  partXhodo = eventTBVertex.x();
  partYhodo = eventTBVertex.y();

  product->setBeamPosition(partXhodo, partYhodo);

  // Asynchronous phase shift
  double thisPhaseShift = flatDistribution_->fire();

  product->setPhaseShift(thisPhaseShift);
  LogDebug("EcalTBInfo") << "Asynchronous Phaseshift = " << thisPhaseShift;

  // store the object in the framework event

  event.put(product);
}


