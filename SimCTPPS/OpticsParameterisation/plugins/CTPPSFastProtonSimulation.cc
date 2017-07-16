/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
 *   Leszek Grzanka
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "CondFormats/CTPPSOpticsObjects/interface/LHCOpticsApproximator.h"

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"

#include "CLHEP/Random/RandGauss.h"

#include <unordered_map>
#include <array>

class CTPPSFastProtonSimulation : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSFastProtonSimulation( const edm::ParameterSet& );
    ~CTPPSFastProtonSimulation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    struct CTPPSPotInfo {
      CTPPSPotInfo() : detid( 0 ), resolution( 0.0 ), z_position( 0.0 ), approximator( 0 ) {}
      CTPPSPotInfo( const CTPPSDetId& det_id, double resol, double z_position, LHCOpticsApproximator* approx ) :
        detid( det_id ), resolution( resol ), z_position( z_position ), approximator( approx ) {}

      CTPPSDetId detid;
      double resolution;
      double z_position;
      LHCOpticsApproximator* approximator;
    };

    virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    void transportProtonTrack( const HepMC::GenParticle*, edm::DetSetVector<TotemRPRecHit>& );
    bool produceHit( const CLHEP::Hep3Vector&, const CTPPSDetId&, TotemRPRecHit& ) const;

    edm::EDGetTokenT<edm::HepMCProduct> protonsToken_;

    edm::ParameterSet beamConditions_;
    double sqrtS_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;

    bool simulateDetectorsResolution_;
    bool roundToPitch_;

    bool checkApertures_;
    bool produceHitsRelativeToBeam_;

    /// strip pitch in mm
    double pitch_;
    /// size of insensitive margin at sensor's edge facing the beam, in mm
    double insensitiveMargin_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::vector<CTPPSPotInfo> pots_;
    std::unordered_map<unsigned int, std::vector<CTPPSDetId> > pots_list_;

    CLHEP::HepRandomEngine* rnd_;

    /// internal variable: v position of strip 0, in mm
    double stripZeroPosition_;

    edm::ESHandle<TotemRPGeometry> geometry_;

    static const bool invertBeamCoordinatesSystem_;
};

const bool CTPPSFastProtonSimulation::invertBeamCoordinatesSystem_ = true;

CTPPSFastProtonSimulation::CTPPSFastProtonSimulation( const edm::ParameterSet& iConfig ) :
  protonsToken_( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  beamConditions_             ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  sqrtS_                      ( beamConditions_.getParameter<double>( "sqrtS" ) ),
  halfCrossingAngleSector45_  ( beamConditions_.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_  ( beamConditions_.getParameter<double>( "halfCrossingAngleSector56" ) ),
  yOffsetSector45_            ( beamConditions_.getParameter<double>( "yOffsetSector45" ) ),
  yOffsetSector56_            ( beamConditions_.getParameter<double>( "yOffsetSector56" ) ),
  simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  roundToPitch_               ( iConfig.getParameter<bool>( "roundToPitch" ) ),
  checkApertures_             ( iConfig.getParameter<bool>( "checkApertures" ) ),
  produceHitsRelativeToBeam_  ( iConfig.getParameter<bool>( "produceHitsRelativeToBeam" ) ),
  pitch_                      ( iConfig.getParameter<double>( "pitch" ) ),
  insensitiveMargin_          ( iConfig.getParameter<double>( "insensitiveMargin" ) ),
  opticsFileBeam1_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorPackages_           ( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) ),
  rnd_( 0 )
{
  produces< edm::DetSetVector<TotemRPRecHit> >();

  // v position of strip 0
  stripZeroPosition_ = RPTopology::last_strip_to_border_dist_ + (RPTopology::no_of_strips_-1)*RPTopology::pitch_ - RPTopology::y_width_/2.;

  auto f_in_optics_beam1 = std::make_unique<TFile>( opticsFileBeam1_.fullPath().c_str() ),
       f_in_optics_beam2 = std::make_unique<TFile>( opticsFileBeam2_.fullPath().c_str() );

  // load optics and interpolators
  for ( const auto& rp : detectorPackages_ ) {
    const std::string interp_name = rp.getParameter<std::string>( "interpolatorName" );
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "potId" );
    const double det_resol = rp.getParameter<double>( "resolution" );
    const double z_position = rp.getParameter<double>( "zPosition" );
    CTPPSDetId detid( raw_detid );

    if ( detid.arm()==0 ) // sector 45 -- beam 2
      pots_.emplace_back( detid, det_resol, z_position, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam2->Get( interp_name.c_str() ) ) );
    if ( detid.arm()==1 ) // sector 56 -- beam 1
      pots_.emplace_back( detid, det_resol, z_position, dynamic_cast<LHCOpticsApproximator*>( f_in_optics_beam1->Get( interp_name.c_str() ) ) );
  }
}

CTPPSFastProtonSimulation::~CTPPSFastProtonSimulation()
{}

void
CTPPSFastProtonSimulation::beginRun( const edm::Run&, const edm::EventSetup& iSetup )
{
  // get geometry
  iSetup.get<VeryForwardMisalignedGeometryRecord>().get( geometry_ );

  for ( const auto& rp : pots_ ) {
    std::vector<CTPPSDetId>& list = pots_list_[rp.detid.rawId()];
    for ( TotemRPGeometry::mapType::const_iterator it=geometry_->beginDet(); it!=geometry_->endDet(); ++it ) {
      const CTPPSDetId detid( it->first );
      if ( detid.getRPId()!=rp.detid ) continue;
      list.push_back( detid );
    }
  }
}

void
CTPPSFastProtonSimulation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  // prepare output
  std::unique_ptr< edm::DetSetVector<TotemRPRecHit> > pRecHits( new edm::DetSetVector<TotemRPRecHit>() );

  if ( simulateDetectorsResolution_ ) {
    edm::Service<edm::RandomNumberGenerator> rng;
    rnd_ = &( rng->getEngine( iEvent.streamID() ) );
  }

  edm::Handle<edm::HepMCProduct> hepmc_prod;
  iEvent.getByToken( protonsToken_, hepmc_prod );
  const HepMC::GenEvent& evt = hepmc_prod->getHepMCData();
  //const HepMC::GenEvent* evt = hepmc_prod->GetEvent();

  // loop over event vertices
  for ( HepMC::GenEvent::vertex_const_iterator vtx=evt.vertices_begin(); vtx!=evt.vertices_end(); ++vtx ) {
    //const HepMC::FourVector& vertex = (*vtx)->position(); // in mm

    // loop over outgoing particles
    for ( HepMC::GenVertex::particles_out_const_iterator p=(*vtx)->particles_out_const_begin(); p!=(*vtx)->particles_out_const_end(); ++p ) {
      // run simulation
      if ( ( *p )->status()!=1 || ( *p )->pdg_id()!=2212 ) continue; // only transport stable protons

      transportProtonTrack( *p, *pRecHits );
      //FIXME add an association map proton track <-> sim hits
    }
  }

  iEvent.put( std::move( pRecHits ) );
}

void
CTPPSFastProtonSimulation::transportProtonTrack( const HepMC::GenParticle* in_trk, edm::DetSetVector<TotemRPRecHit>& out_hits )
{
  /// implemented according to LHCOpticsApproximator::Transport_m_GeV
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  // transport the proton into each pot
  for ( const auto& rp : pots_ ) {
    // so far only works for strips
    if ( rp.detid.subdetId()!=CTPPSDetId::sdTrackingStrip ) continue;
    if ( pots_list_.find( rp.detid.rawId() )==pots_list_.end() ) continue;

    // retrieve the sensor from geometry
    for ( const auto& detid : pots_list_[rp.detid.rawId()] ) {

      const HepMC::FourVector mom = in_trk->momentum();

      // first check the side
      if ( detid.arm()==0 && mom.z()<0.0 ) continue;
      if ( detid.arm()==1 && mom.z()>0.0 ) continue;

      edm::DetSet<TotemRPRecHit>& hits = out_hits.find_or_insert( detid );

      // get geometry
      const CLHEP::Hep3Vector gl_o = geometry_->LocalToGlobal( detid, CLHEP::Hep3Vector() ); // in mm
      const double optics_z0 = rp.z_position*1.e3; // in mm

      const HepMC::GenVertex* vtx = in_trk->production_vertex();
      // convert physics kinematics to the LHC reference frame
      double th_x = atan2( mom.x(), mom.z() ), th_y = atan2( mom.y(), mom.z() );
      if ( mom.z()<0.0 ) { th_x = M_PI-th_x; th_y = M_PI-th_y; }
      double vtx_x = vtx->position().x()*1.e-3, vtx_y = vtx->position().y()*1.e-3; // express in metres

      double half_cr_angle = 0.0, vtx_y_offset = 0.0;
      int z_sign = 0;
      if ( detid.arm()==0 ) {
        z_sign = -1; //FIXME LHC or CMS convention?
        half_cr_angle = halfCrossingAngleSector45_;
        vtx_y_offset = yOffsetSector45_;
      }
      if ( detid.arm()==1 ) {
        z_sign = +1;
        half_cr_angle = halfCrossingAngleSector56_;
        vtx_y_offset = yOffsetSector56_;
      }

      const double xi = 1.-mom.e()/sqrtS_*2.0;

      // transport proton to its corresponding RP
      std::array<double,5> kin_in_tr = { { vtx_x, ( th_x + half_cr_angle ) * ( 1.-xi ), vtx_y + vtx_y_offset, th_y * ( 1.-xi ), -xi } }, kin_out_tr;

      bool tr_proton_transported = rp.approximator->Transport( kin_in_tr.data(), kin_out_tr.data(), checkApertures_, invertBeamCoordinatesSystem_ );

      // stop if proton not transportable
      if ( !tr_proton_transported ) return;

      const double a_x_tr = kin_out_tr[1]/( 1.-xi );
      const double a_y_tr = kin_out_tr[3]/( 1.-xi );
      const double b_x_tr = kin_out_tr[0];
      const double b_y_tr = kin_out_tr[2];

      //printf("    track: ax=%f, bx=%f, ay=%f, by=%f\n", a_x_tr, b_x_tr, a_y_tr, b_y_tr);

      // evaluate positions (in mm) of track and beam
      const double de_z = ( gl_o.z()-optics_z0 ) * z_sign;

      const double x_tr = a_x_tr * de_z + b_x_tr * 1.e3;
      const double y_tr = a_y_tr * de_z + b_y_tr * 1.e3;

      // global hit in coordinates "aligned to beam" (as in the RP alignment)
      CLHEP::Hep3Vector h_glo( x_tr, y_tr, gl_o.z() );

      if ( produceHitsRelativeToBeam_ ) {
        std::array<double,5> kin_in_be = { { 0.0, half_cr_angle, vtx_y_offset, 0.0, 0.0 } }, kin_out_be;

        bool be_proton_transported = rp.approximator->Transport( kin_in_be.data(), kin_out_be.data(), checkApertures_, invertBeamCoordinatesSystem_ );

        // stop if proton not transportable
        if ( be_proton_transported ) {
          const double a_x_be = kin_out_be[1];
          const double a_y_be = kin_out_be[3];
          const double b_x_be = kin_out_be[0];
          const double b_y_be = kin_out_be[2];

          //printf("    beam: ax=%f, bx=%f, ay=%f, by=%f\n", a_x_be, b_x_be, a_y_be, b_y_be);

          const double x_be = a_x_be * de_z + b_x_be * 1.e3;
          const double y_be = a_y_be * de_z + b_y_be * 1.e3;

          /*std::cout << detid << ", z = " << gl_o.z() << ", de z = " << (gl_o.z() - optics_z0) <<
            " | track: x=" << x_tr << ", y=" << y_tr <<
            " | beam: x=" << x_be << ", y=" << y_be <<
            std::endl;*/

          h_glo -= CLHEP::Hep3Vector( x_be, y_be, 0.0 );
        }
      }
      //std::cout << detid << ", z = " << gl_o.z() << ", de z = " << (gl_o.z() - optics_z0) << " | track: x=" << x_tr << ", y=" << y_tr << std::endl;

      TotemRPRecHit hit; // all coordinates in mm
      if ( produceHit( h_glo, detid, hit ) ) hits.push_back( hit );
    }
    /*const double rp_resol = ( simulateDetectorsResolution_ ) ? rp.resolution : 0.0;

    // simulate detector resolution
    kin_out[0] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_x
    kin_out[2] += CLHEP::RandGauss::shoot( rnd_ ) * rp_resol; // vtx_y*/
  }
}

bool       
CTPPSFastProtonSimulation::produceHit( const CLHEP::Hep3Vector& coord_global, const CTPPSDetId& detid, TotemRPRecHit& rechit ) const
{
  // transform hit global to local coordinates
  const CLHEP::Hep3Vector h_loc = geometry_->GlobalToLocal( detid, coord_global );

  double u = h_loc.x();
  double v = h_loc.y();

  // is it within detector?
  if ( !RPTopology::IsHit( u, v, insensitiveMargin_ ) ) return false;

  // round the measurement
  if ( roundToPitch_ ) {
    double m = stripZeroPosition_ - v;
    int strip = static_cast<int>( floor( m/pitch_ + 0.5 ) );
    v = stripZeroPosition_ - pitch_ * strip;
  }

  const double sigma = pitch_ / sqrt( 12. );

  rechit = TotemRPRecHit( v, sigma );
  return true;
}

void
CTPPSFastProtonSimulation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSFastProtonSimulation );

