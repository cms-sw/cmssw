/*
 * \file EcalTBMCInfoProducer.cc
 *
 * $Id: EcalTBMCInfoProducer.cc,v 1.1 2006/05/30 16:51:53 fabiocos Exp $
 *
*/

#include "SimG4CMS/EcalTestBeam/interface/EcalTBMCInfoProducer.h"

EcalTBMCInfoProducer::EcalTBMCInfoProducer(const edm::ParameterSet& ps) {
  
  produces<PEcalTBInfo>();

  string CrystalMapFile = ps.getUntrackedParameter<string>("CrystalMapFile","BarrelSM1CrystalCenterElectron120GeV.dat");
  GenVtxLabel = ps.getUntrackedParameter<string>("moduleLabelVtx","VtxSmeared");
  double fMinEta = ps.getUntrackedParameter<double>("MinEta");
  double fMaxEta = ps.getUntrackedParameter<double>("MaxEta");
  double fMinPhi = ps.getUntrackedParameter<double>("MinPhi");
  double fMaxPhi = ps.getUntrackedParameter<double>("MaxPhi");
  beamEta = (fMaxEta+fMinEta)/2.;
  beamPhi = (fMaxPhi+fMinPhi)/2.;
  beamTheta = 2.0*atan(exp(-beamEta));
  beamXoff = ps.getUntrackedParameter<double>("BeamMeanX");
  beamYoff = ps.getUntrackedParameter<double>("BeamMeanX");
   
  theTestMap = new EcalTBCrystalMap(CrystalMapFile);
  crysNumber = theTestMap->CrystalIndex(beamEta, beamPhi);

  edm::LogInfo("EcalTBInfo") << "Initialize TB MC ECAL info producer with parameters: \n"
                             << "Crystal map file:  " << CrystalMapFile << "\n"
                             << "Beam average eta = " << beamEta << "\n"
                             << "Beam average phi = " << beamPhi << "\n"
                             << "Corresponding to crystal number = " << crysNumber << "\n"
                             << "Beam X offset =    " << beamXoff << "\n"
                             << "Beam Y offset =    " << beamYoff;

  // rotation matrix to move from the CMS reference frame to the test beam one
  // find the axis orthogonal to the plane spanned by z and the vector

  double xx = sin(beamTheta)*cos(beamPhi);
  double yy = sin(beamTheta)*sin(beamPhi);
  double zz = cos(beamTheta);
  Hep3Vector myDir(xx,yy,zz);
  Hep3Vector zAxis(0.,0.,1.);
  Hep3Vector ortho = myDir.cross(zAxis);

  fromCMStoTB = new HepRotation(ortho, beamTheta);

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
   
  // store the object in the framework event

  event.put(product);
}


