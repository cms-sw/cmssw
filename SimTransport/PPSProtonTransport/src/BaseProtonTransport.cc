#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>

BaseProtonTransport::BaseProtonTransport(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getParameter<bool>("Verbosity")),
      bApplyZShift_(iConfig.getParameter<bool>("ApplyZShift")),
      useBeamPositionFromLHCInfo_(iConfig.getParameter<bool>("useBeamPositionFromLHCInfo")),
      produceHitsRelativeToBeam_(iConfig.getParameter<bool>("produceHitsRelativeToBeam")),
      fPPSRegionStart_45_(iConfig.getParameter<double>("PPSRegionStart_45")),
      fPPSRegionStart_56_(iConfig.getParameter<double>("PPSRegionStart_56")),
      etaCut_(iConfig.getParameter<double>("EtaCut")),
      momentumCut_(iConfig.getParameter<double>("MomentumCut")),
      beamEnergy_(iConfig.getParameter<double>("BeamEnergy")) {
  beamMomentum_ = std::sqrt(beamEnergy_ * beamEnergy_ - ProtonMassSQ);
}

BaseProtonTransport::~BaseProtonTransport() { clear(); }

void BaseProtonTransport::ApplyBeamCorrection(HepMC::GenParticle* p) {
  TLorentzVector p_out;
  p_out.SetPx(p->momentum().px());
  p_out.SetPy(p->momentum().py());
  p_out.SetPz(p->momentum().pz());
  p_out.SetE(p->momentum().e());
  ApplyBeamCorrection(p_out);
  p->set_momentum(HepMC::FourVector(p_out.Px(), p_out.Py(), p_out.Pz(), p_out.E()));
}

void BaseProtonTransport::ApplyBeamCorrection(TLorentzVector& p_out) {
  double thetax = atan(p_out.Px() / fabs(p_out.Pz()));
  double thetay = atan(p_out.Py() / fabs(p_out.Pz()));
  double energy = p_out.E();
  double urad = 1e-6;

  int direction = (p_out.Pz() > 0) ? 1 : -1;

  if (MODE == TransportMode::TOTEM)
    thetax += (p_out.Pz() > 0) ? fCrossingAngleX_45_ * urad : fCrossingAngleX_56_ * urad;

  double dtheta_x = (m_sigmaSTX <= 0.0) ? 0.0 : CLHEP::RandGauss::shoot(engine_, 0., m_sigmaSTX);
  double dtheta_y = (m_sigmaSTY <= 0.0) ? 0.0 : CLHEP::RandGauss::shoot(engine_, 0., m_sigmaSTY);
  double denergy = (m_sig_E <= 0.0) ? 0.0 : CLHEP::RandGauss::shoot(engine_, 0., m_sig_E);

  double s_theta = std::sqrt(pow(thetax + dtheta_x * urad, 2) + std::pow(thetay + dtheta_y * urad, 2));
  double s_phi = std::atan2(thetay + dtheta_y * urad, thetax + dtheta_x * urad);
  energy += denergy;
  double p = std::sqrt(std::pow(energy, 2) - ProtonMassSQ);
  double sint = std::sin(s_theta);

  p_out.SetPx(p * sint * std::cos(s_phi));
  p_out.SetPy(p * sint * std::sin(s_phi));
  p_out.SetPz(p * std::cos(s_theta) * direction);
  p_out.SetE(energy);
}

void BaseProtonTransport::addPartToHepMC(const HepMC::GenEvent* in_evt, HepMC::GenEvent* evt) {
  m_CorrespondenceMap.clear();

  int direction = 0;
  HepMC::GenParticle* gpart;

  unsigned int line;

  for (auto const& it : m_beamPart) {
    line = (it).first;
    gpart = in_evt->barcode_to_particle(line);

    direction = (gpart->momentum().pz() > 0) ? 1 : -1;

    // Totem uses negative Z for sector 56 while Hector uses always positive distance
    double ddd = (direction > 0) ? fPPSRegionStart_45_ : fabs(fPPSRegionStart_56_);

    double time = (ddd * meter - gpart->production_vertex()->position().z() * mm);  // mm

    //
    // ATTENTION: at this point, the vertex at PPS is already in mm
    //
    if (ddd == 0.)
      continue;

    if (verbosity_) {
      LogDebug("BaseProtonTransportEventProcessing")
          << "BaseProtonTransport:: x= " << (*(m_xAtTrPoint.find(line))).second << "\n"
          << "BaseProtonTransport:: y= " << (*(m_yAtTrPoint.find(line))).second << "\n"
          << "BaseProtonTransport:: z= " << ddd * direction * m_to_mm << "\n"
          << "BaseProtonTransport:: t= " << time;
    }
    TLorentzVector const& p_out = (it).second;

    HepMC::GenVertex* vert = new HepMC::GenVertex(HepMC::FourVector((*(m_xAtTrPoint.find(line))).second,
                                                                    (*(m_yAtTrPoint.find(line))).second,
                                                                    ddd * direction * m_to_mm,
                                                                    time + time * 0.001));

    vert->add_particle_out(new HepMC::GenParticle(
        HepMC::FourVector(p_out.Px(), p_out.Py(), p_out.Pz(), p_out.E()), gpart->pdg_id(), 1, gpart->flow()));
    evt->add_vertex(vert);

    int ingoing = 0;  //do not attach the incoming proton to this vertex to avoid duplicating data
    int outgoing = (*vert->particles_out_const_begin())->barcode();

    LHCTransportLink theLink(ingoing, outgoing);
    if (verbosity_)
      LogDebug("BaseProtonTransportEventProcessing")
          << "BaseProtonTransport:addPartToHepMC: LHCTransportLink " << theLink;
    m_CorrespondenceMap.push_back(theLink);
  }
}
void BaseProtonTransport::clear() {
  m_beamPart.clear();
  m_xAtTrPoint.clear();
  m_yAtTrPoint.clear();
  m_CorrespondenceMap.clear();
}
