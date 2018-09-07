#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <cmath>

TotemTransport::TotemTransport():ProtonTransport(){MODE=TOTEM;};
TotemTransport::~TotemTransport()
{
      this->clear();
}
TotemTransport::TotemTransport(const edm::ParameterSet & iConfig, bool verbosity):ProtonTransport(),
        parameters(iConfig.getParameter<edm::ParameterSet>("BeamProtTransportSetup")),
        verbosity(iConfig.getParameter<bool>("Verbosity")),
        model_root_file_r(parameters.getParameter<std::string>("ModelRootFile_R")),
        model_root_file_l(parameters.getParameter<std::string>("ModelRootFile_L")),
        model_ip_150_r_name(parameters.getParameter<std::string>("Model_IP_150_R_Name")),
        model_ip_150_l_name(parameters.getParameter<std::string>("Model_IP_150_L_Name")),
        model_ip_150_r_zmin(parameters.getParameter<double>("Model_IP_150_R_Zmin")),
        model_ip_150_r_zmax(parameters.getParameter<double>("Model_IP_150_R_Zmax")),
        model_ip_150_l_zmin(parameters.getParameter<double>("Model_IP_150_L_Zmin")),
        model_ip_150_l_zmax(parameters.getParameter<double>("Model_IP_150_L_Zmax")), 
        beampipe_aperture_radius(parameters.getParameter<double>("BeampipeApertureRadius"))
{
        fBeamEnergy= parameters.getParameter<double>("sqrtS");
        m_sigmaSTX = parameters.getParameter<double>("beamDivergenceX");
        m_sigmaSTY = parameters.getParameter<double>("beamDivergenceY");
        m_sig_E    = parameters.getParameter<double>("beamEnergyDispersion");
        fCrossingAngle_45 = parameters.getParameter<double>("halfCrossingAngleSector45");
        fCrossingAngle_56 = parameters.getParameter<double>("halfCrossingAngleSector56");
        fVtxMeanX       = iConfig.getParameter<double>("VtxMeanX");
        fVtxMeanY       = iConfig.getParameter<double>("VtxMeanY");
        fVtxMeanZ       = iConfig.getParameter<double>("VtxMeanZ");
        fBeamXatIP      = parameters.getUntrackedParameter<double>("BeamXatIP",fVtxMeanX);
        fBeamYatIP      = parameters.getUntrackedParameter<double>("BeamYatIP",fVtxMeanY);
        bApplyZShift    = parameters.getParameter<bool>("ApplyZShift");

        MODE = TOTEM;

        fBeamMomentum        = sqrt(fBeamEnergy*fBeamEnergy - pow(proton_mass_c2/GeV,2));
/*
        PPSTools::fBeamMomentum=fBeamMomentum;
        PPSTools::fBeamEnergy=fBeamEnergy;
        PPSTools::fCrossingAngleBeam1=fCrossingAngle_56;
        PPSTools::fCrossingAngleBeam2=fCrossingAngle_45;
*/
        fPPSRegionStart_56=model_ip_150_r_zmax;
        fPPSRegionStart_45=model_ip_150_l_zmax;
        
        aprox_ip_150_r = ReadParameterization(model_ip_150_r_name,model_root_file_r);
        aprox_ip_150_l = ReadParameterization(model_ip_150_l_name,model_root_file_l);

        if (aprox_ip_150_r == nullptr || aprox_ip_150_l == nullptr) {
                edm::LogError("TotemTransport") << "Parameterisation "
                        << model_ip_150_r_name << " or " << model_ip_150_l_name << " missing in file. Cannot proceed. ";
                exit(1);
        }
        edm::LogInfo("TotemRPProtonTransportSetup") <<
                "Parameterizations read from file, pointers:" << aprox_ip_150_r << " " << aprox_ip_150_l << " ";
}
void TotemTransport::process(const HepMC::GenEvent * evt , const edm::EventSetup& iSetup, CLHEP::HepRandomEngine * _engine )
{
        engine=_engine;

        for (HepMC::GenEvent::particle_const_iterator eventParticle =evt->particles_begin(); eventParticle != evt->particles_end(); ++eventParticle ) {
         if (!((*eventParticle)->status() == 1 && (*eventParticle)->pdg_id()==2212 )) continue;
         //if (!(fabs((*eventParticle)->momentum().eta())>fEtacut && fabs((*eventParticle)->momentum().pz())>fMomentumMin)) continue;
         unsigned int line = (*eventParticle)->barcode();
         HepMC::GenParticle * gpart = (*eventParticle);
         if ( gpart->pdg_id()!=2212 ) continue; // only transport stable protons
         if ( gpart->status()!=1 /*&& gpart->status()<83 */) continue;
         if ( m_beamPart.find(line) != m_beamPart.end() ) continue;


         transportProton(gpart);
 
         //transportProtonTrack( part, *pRecHits, out_vtx );
         //vtx->set_position( out_vtx );
     }
     addPartToHepMC(const_cast<HepMC::GenEvent*>(evt));
}
bool TotemTransport::transportProton( const HepMC::GenParticle* in_trk)
{
     //PPSTools::LorentzBoost(*(const_cast<HepMC::GenParticle*>(in_trk)),"LAB");
     ApplyBeamCorrection(const_cast<HepMC::GenParticle*>(in_trk));

     const HepMC::GenVertex* in_pos = in_trk->production_vertex();
     const HepMC::FourVector in_mom = in_trk->momentum();
//
// ATTENTION: HepMC uses mm, vertex config of CMS uses cm and SimTransport uses mm
//
     double in_position[3] = {(in_pos->position().x()-fVtxMeanX*cm) / meter+fBeamXatIP*mm/meter,
                              (in_pos->position().y()-fVtxMeanY*cm) / meter+fBeamYatIP*mm/meter,
                              (in_pos->position().z()-fVtxMeanZ*cm) / meter};  // move to z=0 if configured below
                              //(in_pos->position().y()-fVtxMeanY*cm) / meter+fBeamYatIP*mm/meter, Zin_};  // CHECK! starting Z was at 0

// (bApplyZShift) -- The TOTEM parameterization requires the shift to z=0
     double fCrossingAngle = (in_mom.z()>0)?fCrossingAngle_45:-fCrossingAngle_56;
     in_position[0] = in_position[0]+(tan((long double)fCrossingAngle*urad)-((long double)in_mom.x())/((long double)in_mom.z()))*in_position[2];
     in_position[1] = in_position[1]-((long double)in_mom.y())/((long double)in_mom.z())*in_position[2];
     in_position[2] = 0.;
//
     double in_momentum[3] = {in_mom.x(), in_mom.y() , in_mom.z()};
     double out_position[3];
     double out_momentum[3];
     edm::LogInfo("TotemTransport") << "before transport ->" <<
             " position: " << in_position[0] << ", " << in_position[1] << ", " << in_position[2] <<
             " momentum: " << in_momentum[0] << ", " << in_momentum[1] << ", " << in_momentum[2];

     LHCOpticsApproximator* approximator_= nullptr;
     if (in_mom.z()>0) {approximator_ = aprox_ip_150_l; Zin_ = model_ip_150_l_zmin; Zout_ = model_ip_150_l_zmax;}
     else              {approximator_ = aprox_ip_150_r; Zin_ = model_ip_150_r_zmin; Zout_ = model_ip_150_r_zmax;}

     bool invert_beam_coord_system=true; // it doesn't matter the option here, it is hard coded as TRUE inside LHCOpticsApproximator!

     bool tracked = approximator_->Transport_m_GeV(in_position, in_momentum, out_position, out_momentum, invert_beam_coord_system, Zout_ - Zin_);

     if (!tracked) return false;

     edm::LogInfo("TotemTransport") << "after transport -> " <<
             "position: " << out_position[0] << ", " << out_position[1] << ", " << out_position[2] <<
             "momentum: " << out_momentum[0] << ", " << out_momentum[1] << ", " << out_momentum[2];

     if (out_position[0] * out_position[0] + out_position[1] * out_position[1] >
                     beampipe_aperture_radius * beampipe_aperture_radius) {
             edm::LogInfo("TotemTransport") << "Proton ouside beampipe";
             edm::LogInfo("TotemTransport") << "===== END Transport " << "====================";
             return false;
     }

    CLHEP::Hep3Vector out_pos(out_position[0] * meter, out_position[1] * meter, out_position[2] * meter);
    CLHEP::Hep3Vector out_mom(out_momentum[0], out_momentum[1], out_momentum[2]);
    edm::LogInfo("TotemRPProtonTransportModel") << "output -> " <<
    "position: " << out_pos << " momentum: " << out_mom << std::endl;
    //int Direction = (in_mom.z()>0)?1:-1;
     double px = -out_momentum[0];
     double py = out_momentum[1];  // this need to be checked again, since it seems an invertion is occuring in  the prop.
     double pz = out_momentum[2];
     double e = sqrt(px*px+py*py+pz*pz+pow(CLHEP::proton_mass_c2/GeV,2));
     CLHEP::HepLorentzVector* p_out = new CLHEP::HepLorentzVector(px,py,pz,e);
     double x1_ctpps = -out_position[0]*meter; // Totem parameterization uses meter, one need it in millimeter
     double y1_ctpps = -out_position[1]*meter;

     unsigned int line = in_trk->barcode();

     if(m_verbosity) LogDebug("HectorTransportEventProcessing") <<
             "HectorTransport:filterPPS: barcode = " << line << " x=  "<< x1_ctpps <<" y= " << y1_ctpps;

     m_beamPart[line]    = p_out;
     m_xAtTrPoint[line]  = x1_ctpps;
     m_yAtTrPoint[line]  = y1_ctpps;
     return true;
}
LHCOpticsApproximator* TotemTransport::ReadParameterization(const std::string& model_name, const std::string& rootfile)
{
    edm::FileInPath fileName(rootfile.c_str());
    TFile *f = TFile::Open(fileName.fullPath().c_str(), "read");
    if (!f) {
        edm::LogError("TotemRPProtonTransportSetup") << "File " << fileName << " not found. Exiting.";
        return nullptr;
    }
    edm::LogInfo("TotemRPProtonTransportSetup") << "Root file opened, pointer:" << f;

    // read parametrization
    LHCOpticsApproximator* aprox = (LHCOpticsApproximator *) f->Get(model_name.c_str());
    f->Close();
    return aprox;
}
