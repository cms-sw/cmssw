/*
 * \file EcalTBMCInfoProducer.cc
 *
 * $Id: EcalTBMCInfoProducer.cc,v 1.5 2006/10/25 16:54:14 fabiocos Exp $
 *
*/

#include "SimG4CMS/EcalTestBeam/interface/EcalTBMCInfoProducer.h"
#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace cms;

EcalTBMCInfoProducer::EcalTBMCInfoProducer(const edm::ParameterSet& ps) {
  
  produces<PEcalTBInfo>();

  edm::FileInPath CrystalMapFile = ps.getParameter<edm::FileInPath>("CrystalMapFile");
  GenVtxLabel = ps.getUntrackedParameter<string>("moduleLabelVtx","VtxSmeared");
  double fMinEta = ps.getUntrackedParameter<double>("MinEta");
  double fMaxEta = ps.getUntrackedParameter<double>("MaxEta");
  double fMinPhi = ps.getUntrackedParameter<double>("MinPhi");
  double fMaxPhi = ps.getUntrackedParameter<double>("MaxPhi");
  beamEta = (fMaxEta+fMinEta)/2.;
  beamPhi = (fMaxPhi+fMinPhi)/2.;
  beamTheta = 2.0*atan(exp(-beamEta));
  beamXoff = ps.getUntrackedParameter<double>("BeamMeanX",0.0);
  beamYoff = ps.getUntrackedParameter<double>("BeamMeanX",0.0);
   
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
  }

  edm::LogInfo("EcalTBInfo") << "Initialize TB MC ECAL info producer with parameters: \n"
                             << "Crystal map file:  " << CrystalMapFile << "\n"
                             << "Beam average eta = " << beamEta << "\n"
                             << "Beam average phi = " << beamPhi << "\n"
                             << "Corresponding to crystal number = " << crysNumber << "\n"
                             << "Beam X offset =    " << beamXoff << "\n"
                             << "Beam Y offset =    " << beamYoff;

  // rotation matrix to move from the CMS reference frame to the test beam one

  fromCMStoTB = new HepRotation();

  double xx = -cos(beamTheta)*cos(beamPhi);
  double xy = -cos(beamTheta)*sin(beamPhi);
  double xz = sin(beamTheta);
  
  double yx = sin(beamPhi);
  double yy = -cos(beamPhi);
  double yz = 0.;
  
  double zx = sin(beamTheta)*cos(beamPhi);
  double zy = sin(beamTheta)*sin(beamPhi);
  double zz = cos(beamTheta);

  const HepRep3x3 mCMStoTB(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  fromCMStoTB->set(mCMStoTB);

}
 
EcalTBMCInfoProducer::~EcalTBMCInfoProducer() {

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

  Hep3Vector eventCMSVertex = (*Vtx)->position().v();

  LogDebug("EcalTBInfo") << "Generated vertex position = " 
                         << eventCMSVertex.x() << " " 
                         << eventCMSVertex.y() << " " 
                         << eventCMSVertex.z();

  Hep3Vector & eventTBVertex = eventCMSVertex.transform((*fromCMStoTB));

  LogDebug("EcalTBInfo") << "Rotated vertex position   = "
                         << eventTBVertex.x() << " " 
                         << eventTBVertex.y() << " " 
                         << eventTBVertex.z();

  partXhodo = eventTBVertex.x();
  partYhodo = eventTBVertex.y();

  product->setBeamPosition(partXhodo, partYhodo);

  // Asynchronous phase shift
  double thisPhaseShift = RandFlat::shoot();
  product->setPhaseShift(thisPhaseShift);
  LogDebug("EcalTBInfo") << "Asynchronous Phaseshift = " << thisPhaseShift;

  // store the object in the framework event

  event.put(product);
}


