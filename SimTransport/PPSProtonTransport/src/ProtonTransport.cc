#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>

ProtonTransport::ProtonTransport(){};
ProtonTransport::~ProtonTransport(){};
void ProtonTransport::clear() {
  m_beamPart.clear();
  m_xAtTrPoint.clear();
  m_yAtTrPoint.clear();
};

void ProtonTransport::addPartToHepMC(HepMC::GenEvent* evt) {
  NEvent++;
  m_CorrespondenceMap.clear();

  int direction = 0;
  HepMC::GenParticle* gpart;

  unsigned int line;

  for (auto const& it : m_beamPart) {
    line = (it).first;
    gpart = evt->barcode_to_particle(line);

    direction = (gpart->momentum().pz() > 0) ? 1 : -1;

    double ddd =
        (direction > 0)
            ? fPPSRegionStart_45
            : fabs(
                  fPPSRegionStart_56);  // Totem uses negative Z for sector 56 while Hector uses always positive distance

    double time = (ddd * meter - gpart->production_vertex()->position().z() * mm);  // mm

    //
    // ATTENTION: at this point, the vertex at PPS is already in mm
    //
    if (ddd == 0.)
      continue;
    if (m_verbosity) {
      LogDebug("ProtonTransportEventProcessing")
          << "ProtonTransport:: x= " << (*(m_xAtTrPoint.find(line))).second << "\n"
          << "ProtonTransport:: y= " << (*(m_yAtTrPoint.find(line))).second << "\n"
          << "ProtonTransport:: z= " << ddd * direction * m_to_mm << "\n"
          << "ProtonTransport:: t= " << time;
    }
    TLorentzVector const& p_out = (it).second;

    HepMC::GenVertex* vert = new HepMC::GenVertex(HepMC::FourVector((*(m_xAtTrPoint.find(line))).second,
                                                                    (*(m_yAtTrPoint.find(line))).second,
                                                                    ddd * direction * m_to_mm,
                                                                    time + time * 0.001));

    gpart->set_status(2);
    vert->add_particle_in(gpart);
    vert->add_particle_out(new HepMC::GenParticle(
        HepMC::FourVector(p_out.Px(), p_out.Py(), p_out.Pz(), p_out.E()), gpart->pdg_id(), 1, gpart->flow()));
    evt->add_vertex(vert);

    int ingoing = (*vert->particles_in_const_begin())->barcode();
    int outgoing = (*vert->particles_out_const_begin())->barcode();

    LHCTransportLink theLink(ingoing, outgoing);
    if (m_verbosity)
      LogDebug("ProtonTransportEventProcessing") << "ProtonTransport:addPartToHepMC: LHCTransportLink " << theLink;
    m_CorrespondenceMap.push_back(theLink);
  }
}
void ProtonTransport::ApplyBeamCorrection(HepMC::GenParticle* p) {
  TLorentzVector p_out;
  p_out.SetPx(p->momentum().px());
  p_out.SetPy(p->momentum().py());
  p_out.SetPz(p->momentum().pz());
  p_out.SetE(p->momentum().e());
  ApplyBeamCorrection(p_out);
  p->set_momentum(HepMC::FourVector(p_out.Px(), p_out.Py(), p_out.Pz(), p_out.E()));
}
void ProtonTransport::ApplyBeamCorrection(TLorentzVector& p_out) {
  double theta = p_out.Theta();
  double thetax = atan(p_out.Px() / fabs(p_out.Pz()));
  double thetay = atan(p_out.Py() / fabs(p_out.Pz()));
  double energy = p_out.E();
  double urad = 1e-6;

  int direction = (p_out.Pz() > 0) ? 1 : -1;

  if (p_out.Pz() < 0)
    theta = TMath::Pi() - theta;

  if (MODE == TransportMode::TOTEM)
    thetax += (p_out.Pz() > 0) ? fCrossingAngle_45 * urad : fCrossingAngle_56 * urad;

  double dtheta_x = (double)CLHEP::RandGauss::shoot(engine, 0., m_sigmaSTX);
  double dtheta_y = (double)CLHEP::RandGauss::shoot(engine, 0., m_sigmaSTY);
  double denergy = (double)CLHEP::RandGauss::shoot(engine, 0., m_sig_E);

  double s_theta = sqrt(pow(thetax + dtheta_x * urad, 2) + pow(thetay + dtheta_y * urad, 2));
  double s_phi = atan2(thetay + dtheta_y * urad, thetax + dtheta_x * urad);
  energy += denergy;
  double p = sqrt(pow(energy, 2) - ProtonMassSQ);

  p_out.SetPx((double)p * sin(s_theta) * cos(s_phi));
  p_out.SetPy((double)p * sin(s_theta) * sin(s_phi));
  p_out.SetPz((double)p * (cos(s_theta)) * direction);
  p_out.SetE(energy);
}
