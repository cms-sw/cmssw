#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"

ProtonTransport::ProtonTransport() {};
ProtonTransport::~ProtonTransport() {};
void ProtonTransport::addPartToHepMC( HepMC::GenEvent * evt )
{
    NEvent++;
    theCorrespondenceMap.clear();
    std::map< unsigned int, CLHEP::HepLorentzVector* >::iterator it;

    int direction=0;
    HepMC::GenParticle * gpart;

    unsigned int line;

    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
        line = (*it ).first;
        gpart = evt->barcode_to_particle( line );

        direction=(gpart->momentum().pz()>0)?1:-1;

        double ddd=(direction>0)?fPPSRegionStart_45:fabs(fPPSRegionStart_56); // Totem uses negative Z for sector 56 while Hector uses always positive distance

        double time = ( ddd*meter - gpart->production_vertex()->position().z()*mm ); // mm

//
// ATTENTION: at this point, the vertex at PPS is already in mm
//
        if(ddd == 0.) continue;
        if(m_verbosity) {
                LogDebug("HectorTransportEventProcessing") <<"HectorTransport:: x= "<< (*(m_xAtTrPoint.find(line))).second<< "\n"
                        <<"HectorTransport:: y= "<< (*(m_yAtTrPoint.find(line))).second<< "\n"
                        <<"HectorTransport:: z= "<< ddd * direction*m_to_mm << "\n"
                        <<"HectorTransport:: t= "<< time;
        }
        CLHEP::HepLorentzVector* p_out = (*it).second;

        HepMC::GenVertex * vert = new HepMC::GenVertex(
                        HepMC::FourVector( (*(m_xAtTrPoint.find(line))).second,
                                (*(m_yAtTrPoint.find(line))).second,
                                ddd * direction *m_to_mm, time + time*0.001 ) );

        gpart->set_status( 2 );
        vert->add_particle_in( gpart );
        vert->add_particle_out( new HepMC::GenParticle( HepMC::FourVector(p_out->px(),p_out->py(),p_out->pz(),p_out->e()), gpart->pdg_id(), 1, gpart->flow() )) ;
        evt->add_vertex( vert );

        int ingoing = (*vert->particles_in_const_begin())->barcode();
        int outgoing = (*vert->particles_out_const_begin())->barcode();

        LHCTransportLink theLink(ingoing,outgoing);
        if (m_verbosity) LogDebug("HectorTransportEventProcessing") << "HectorTransport:addPartToHepMC: LHCTransportLink " << theLink;
        theCorrespondenceMap.push_back(theLink);
    }
}
void ProtonTransport::ApplyBeamCorrection(HepMC::GenParticle* p)
{
     CLHEP::HepLorentzVector p_out;
     p_out.setPx(p->momentum().px());
     p_out.setPy(p->momentum().py());
     p_out.setPz(p->momentum().pz());
     p_out.setE(p->momentum().e());
     ApplyBeamCorrection(p_out);
     p->set_momentum(HepMC::FourVector(p_out.px(),p_out.py(),p_out.pz(),p_out.e()));
}
void ProtonTransport::ApplyBeamCorrection(CLHEP::HepLorentzVector& p_out)
{
    double theta  = p_out.theta();
    double thetax = atan(p_out.px()/fabs(p_out.pz()));
    double thetay = atan(p_out.py()/fabs(p_out.pz()));
    double energy = p_out.e();
    double urad = 1e-6;

    int direction = (p_out.pz()>0)?1:-1;

    if (p_out.pz()<0) theta=CLHEP::pi-theta;

    if (MODE==TOTEM) thetax+=(p_out.pz()>0)?fCrossingAngle_45*urad:fCrossingAngle_56*urad;

    double dtheta_x = (double)CLHEP::RandGauss::shoot(engine,0.,m_sigmaSTX);
    double dtheta_y = (double)CLHEP::RandGauss::shoot(engine,0.,m_sigmaSTY);
    double denergy  = (double)CLHEP::RandGauss::shoot(engine,0.,m_sig_E);

    double s_theta = sqrt(pow(thetax+dtheta_x*urad,2)+pow(thetay+dtheta_y*urad,2));
    double s_phi = atan2(thetay+dtheta_y*urad,thetax+dtheta_x*urad);
    energy+=denergy;
    double p = sqrt(pow(energy,2)-pow(CLHEP::proton_mass_c2/GeV,2));

    p_out.setPx((double)p*sin(s_theta)*cos(s_phi));
    p_out.setPy((double)p*sin(s_theta)*sin(s_phi));
    p_out.setPz((double)p*(cos(s_theta))*direction);
    p_out.setE(energy);
}
