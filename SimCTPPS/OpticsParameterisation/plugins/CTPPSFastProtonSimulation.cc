/****************************************************************************
 *
 * This is a part of CTPPS offline software
 * Authors:
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
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

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
    //void transportProtonTrack( const HepMC::GenParticle*, edm::DetSetVector<TotemRPRecHit>&, HepMC::FourVector& ) const;
    void transportProtonTrack( const HepMC::GenParticle*, std::unordered_map<unsigned int, std::array<double,5> >&, std::unordered_map<unsigned int, std::array<double,5> >&, HepMC::FourVector& ) const;

    //--- TOTEM strips simulation
    void generateRecHits( const CTPPSDetId&, const std::array<double,5>&, const std::array<double,5>&, double, edm::DetSetVector<TotemRPRecHit>& ) const;
    bool produceHit( const CLHEP::Hep3Vector&, TotemRPRecHit& ) const;

    //--- scoring plane
    bool produceTrack( const CTPPSDetId&, const std::array<double,5>& kin_tr, CTPPSLocalTrackLite&, double, double ) const;

    edm::EDGetTokenT<edm::HepMCProduct> protonsToken_;

    double sqrtS_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    double yOffsetSector45_, yOffsetSector56_;

    bool checkApertures_;
    bool produceHitsRelativeToBeam_;

    /// simulate the detectors smearing for the scoring plane?
    bool simulateDetectorsResolution_;

    bool roundToPitch_;
    /// strip pitch in mm
    double pitch_;
    /// size of insensitive margin at sensor's edge facing the beam, in mm
    double insensitiveMargin_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::vector<CTPPSPotInfo> pots_;
    std::unordered_map<unsigned int, std::vector<CTPPSDetId> > strips_list_;

    CLHEP::HepRandomEngine* rnd_;

    /// internal variable: v position of strip 0, in mm
    double stripZeroPosition_;

    edm::ESHandle<TotemRPGeometry> geometry_;

    static const bool invertBeamCoordinatesSystem_;
};

const bool CTPPSFastProtonSimulation::invertBeamCoordinatesSystem_ = true;

CTPPSFastProtonSimulation::CTPPSFastProtonSimulation( const edm::ParameterSet& iConfig ) :
  protonsToken_( consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>( "beamParticlesTag" ) ) ),
  sqrtS_                      ( iConfig.getParameter<double>( "sqrtS" ) ),
  halfCrossingAngleSector45_  ( iConfig.getParameter<double>( "halfCrossingAngleSector45" ) ),
  halfCrossingAngleSector56_  ( iConfig.getParameter<double>( "halfCrossingAngleSector56" ) ),
  yOffsetSector45_            ( iConfig.getParameter<double>( "yOffsetSector45" ) ),
  yOffsetSector56_            ( iConfig.getParameter<double>( "yOffsetSector56" ) ),
  checkApertures_             ( iConfig.getParameter<bool>( "checkApertures" ) ),
  produceHitsRelativeToBeam_  ( iConfig.getParameter<bool>( "produceHitsRelativeToBeam" ) ),
  simulateDetectorsResolution_( iConfig.getParameter<bool>( "simulateDetectorsResolution" ) ),
  roundToPitch_               ( iConfig.getParameter<bool>( "roundToPitch" ) ),
  pitch_                      ( iConfig.getParameter<double>( "pitch" ) ),
  insensitiveMargin_          ( iConfig.getParameter<double>( "insensitiveMargin" ) ),
  opticsFileBeam1_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorPackages_           ( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) ),
  rnd_( 0 ),
  stripZeroPosition_( RPTopology::last_strip_to_border_dist_ + ( RPTopology::no_of_strips_-1 )*RPTopology::pitch_ - RPTopology::y_width_*0.5 ) // v position of strip 0
{
  produces<edm::HepMCProduct>( "smeared" );
  produces<edm::DetSetVector<TotemRPRecHit> >();
  produces<std::vector<CTPPSLocalTrackLite> >( "scoringPlane" );

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

  std::ostringstream os;
  for ( const auto& rp : pots_ ) {
    std::vector<CTPPSDetId>& list = strips_list_[rp.detid];
    os << "\npot " << rp.detid << ":";
    for ( TotemRPGeometry::mapType::const_iterator it=geometry_->beginDet(); it!=geometry_->endDet(); ++it ) {
      const CTPPSDetId detid( it->first );
      if ( detid.getRPId()!=rp.detid ) continue;
      list.push_back( detid );
      os << "\n* " << TotemRPDetId( detid );
    }
  }
  edm::LogWarning("CTPPSFastProtonSimulation::beginRun") << "Hierarchy of DetIds for each pot" << os.str();
}

void
CTPPSFastProtonSimulation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::Handle<edm::HepMCProduct> hepmc_prod;
  iEvent.getByToken( protonsToken_, hepmc_prod );

  edm::Service<edm::RandomNumberGenerator> rng;
  rnd_ = &( rng->getEngine( iEvent.streamID() ) );

  // prepare outputs
  std::unique_ptr<edm::DetSetVector<TotemRPRecHit> > pRecHits( new edm::DetSetVector<TotemRPRecHit>() );
  std::unique_ptr<std::vector<CTPPSLocalTrackLite> > pLiteTrks( new std::vector<CTPPSLocalTrackLite>() );

  auto evt = new HepMC::GenEvent( *hepmc_prod->GetEvent() );
  std::unique_ptr<edm::HepMCProduct> pOutProd( new edm::HepMCProduct( evt ) );

  // loop over event vertices
  for ( auto it_vtx=evt->vertices_begin(); it_vtx!=evt->vertices_end(); ++it_vtx ) {
    auto vtx = *( it_vtx );
    //const HepMC::FourVector& vertex = (*vtx)->position(); // in mm

    // loop over outgoing particles
    for ( auto it_part=vtx->particles_out_const_begin(); it_part!=vtx->particles_out_const_end(); ++it_part ) {
      auto part = *( it_part );
      // run simulation
      if ( part->pdg_id()!=2212 ) continue; // only transport stable protons
      if ( part->status()!=1 && part->status()<83 ) continue;

      HepMC::FourVector out_vtx;
      //transportProtonTrack( part, *pRecHits, out_vtx );
      std::unordered_map<unsigned int,std::array<double,5> > m_kin_tr, m_kin_be;
      transportProtonTrack( part, m_kin_tr, m_kin_be, out_vtx );
      for ( const auto& rp : pots_ ) {
        if ( m_kin_tr.find( rp.detid )==m_kin_tr.end() ) continue;
        const double optics_z0 = rp.z_position*1.0e3; // in mm
        generateRecHits( rp.detid, m_kin_tr[rp.detid], m_kin_be[rp.detid], optics_z0, *pRecHits );

        // produce lite local tracks for the scoring plane
        CTPPSLocalTrackLite sp_track;
        const double resol = ( simulateDetectorsResolution_ ) ? rp.resolution : 0.;
        if ( produceTrack( rp.detid, m_kin_tr[rp.detid], sp_track, resol, resol ) ) pLiteTrks->push_back( sp_track );
      }
      vtx->set_position( out_vtx );
      //FIXME add an association map proton track <-> sim hits
    }
  }

  iEvent.put( std::move( pOutProd ), "smeared" );
  iEvent.put( std::move( pRecHits ) );
  iEvent.put( std::move( pLiteTrks ), "scoringPlane" );
}

void
CTPPSFastProtonSimulation::transportProtonTrack( const HepMC::GenParticle* in_trk, std::unordered_map<unsigned int, std::array<double,5> >& kin_out_tr, std::unordered_map<unsigned int, std::array<double,5> >& kin_out_be, HepMC::FourVector& smeared_vtx ) const
{
  /// implemented according to LHCOpticsApproximator::Transport_m_GeV
  /// xi is positive for diffractive protons, thus proton momentum p = (1-xi) * p_nom
  /// horizontal component of proton momentum: p_x = th_x * (1-xi) * p_nom

  const HepMC::GenVertex* vtx = in_trk->production_vertex();
  const HepMC::FourVector mom = in_trk->momentum();
  const double xi = 1.-mom.e()/sqrtS_*2.0;
  double th_x = atan2( mom.px(), fabs( mom.pz() ) );
  double th_y = atan2( mom.py(), fabs( mom.pz() ) );
  while ( th_x<-M_PI ) th_x += 2*M_PI; while ( th_x>+M_PI ) th_x -= 2*M_PI;
  while ( th_y<-M_PI ) th_y += 2*M_PI; while ( th_y>+M_PI ) th_y -= 2*M_PI;
  if ( mom.pz()>0.0 ) { th_x = -th_x; }

  double vtx_x = -vtx->position().x(), vtx_y = vtx->position().y(); // express in metres

  double half_cr_angle = 0.0, vtx_y_offset = 0.0;
  // CMS convention
  if ( mom.z()>0.0 ) { // sector 45
    half_cr_angle = halfCrossingAngleSector45_;
    vtx_y_offset = yOffsetSector45_;
  }
  if ( mom.z()<0.0 ) { // sector 56
    half_cr_angle = halfCrossingAngleSector56_;
    vtx_y_offset = yOffsetSector56_;
  }
  smeared_vtx = vtx->position();
  smeared_vtx.setX( -vtx_x );
  smeared_vtx.setY( vtx_y+vtx_y_offset );

  // transport the proton into each pot
  for ( const auto& rp : pots_ ) {
    // first check the side
    if ( rp.detid.arm()==0 && mom.z()<0.0 ) continue; // sector 45
    if ( rp.detid.arm()==1 && mom.z()>0.0 ) continue; // sector 56

    // transport proton to its corresponding RP
    std::array<double,5> kin_in_tr = { { vtx_x, th_x * ( 1.-xi ), vtx_y + vtx_y_offset, th_y * ( 1.-xi ), -xi } };
    std::array<double,5> kin_in_be = { { 0.0, half_cr_angle, vtx_y_offset, 0.0, 0.0 } };

    if ( !rp.approximator->Transport( kin_in_tr.data(), kin_out_tr[rp.detid].data(), checkApertures_, invertBeamCoordinatesSystem_ ) ) return;
    // stop if proton not transportable

    if ( produceHitsRelativeToBeam_ ) {
      rp.approximator->Transport( kin_in_be.data(), kin_out_be[rp.detid].data(), checkApertures_, invertBeamCoordinatesSystem_ );
    }
  }
}

void
CTPPSFastProtonSimulation::generateRecHits( const CTPPSDetId& detid, const std::array<double,5>& kin_tr, const std::array<double,5>& kin_be, double optics_z0, edm::DetSetVector<TotemRPRecHit>& out_hits ) const
{
  if ( detid.subdetId()!=CTPPSDetId::sdTrackingStrip ) return; // only works with strips

  const double xi = -kin_tr[4];
  const short z_sign = ( detid.arm()==0 ) ? -1 : +1;

  // search for planes
  if ( strips_list_.find( detid )==strips_list_.end() ) return;

  // retrieve the sensor from geometry
  for ( const auto& pl_id : strips_list_.at( detid ) ) {

    edm::DetSet<TotemRPRecHit>& hits = out_hits.find_or_insert( pl_id );

    // get geometry
    const double gl_oz = geometry_->LocalToGlobal( pl_id, CLHEP::Hep3Vector() ).z(); // in mm

    const double a_x_tr = kin_tr[1]/( 1.-xi );
    const double a_y_tr = kin_tr[3]/( 1.-xi );
    const double b_x_tr = kin_tr[0];
    const double b_y_tr = kin_tr[2];

    //printf("    track: ax=%f, bx=%f, ay=%f, by=%f\n", a_x_tr, b_x_tr, a_y_tr, b_y_tr);

    // evaluate positions (in mm) of track and beam
    const double de_z = ( gl_oz-optics_z0 ) * z_sign;

    const double x_tr = a_x_tr * de_z + b_x_tr * 1.0e3;
    const double y_tr = a_y_tr * de_z + b_y_tr * 1.0e3;

    // global hit in coordinates "aligned to beam" (as in the RP alignment)
    CLHEP::Hep3Vector h_glo( x_tr, y_tr, gl_oz );

    if ( produceHitsRelativeToBeam_ ) {
      const double a_x_be = kin_be[1];
      const double a_y_be = kin_be[3];
      const double b_x_be = kin_be[0];
      const double b_y_be = kin_be[2];

      //printf("    beam: ax=%f, bx=%f, ay=%f, by=%f\n", a_x_be, b_x_be, a_y_be, b_y_be);

      const double x_be = a_x_be * de_z + b_x_be * 1.0e3;
      const double y_be = a_y_be * de_z + b_y_be * 1.0e3;

      /*std::cout << pl_id << ", z = " << gl_oz << ", de z = " << (gl_oz - optics_z0) <<
        " | track: x=" << x_tr << ", y=" << y_tr <<
        " | beam: x=" << x_be << ", y=" << y_be <<
        std::endl;*/

      h_glo -= CLHEP::Hep3Vector( x_be, y_be, 0.0 );
    }
    //std::cout << pl_id << ", z = " << gl_oz << ", de z = " << (gl_oz - optics_z0) << " | track: x=" << x_tr << ", y=" << y_tr << std::endl;

    // transform hit global to local coordinates
    const CLHEP::Hep3Vector h_loc = geometry_->GlobalToLocal( pl_id, h_glo );

    TotemRPRecHit hit; // all coordinates in mm
    if ( produceHit( h_loc, hit ) ) hits.push_back( hit );
  }
}

//template<>
bool
CTPPSFastProtonSimulation::produceTrack( const CTPPSDetId& detid, const std::array<double,5>& kin_tr, CTPPSLocalTrackLite& local_track, double rp_resolution_x, double rp_resolution_y ) const
{
  // define track
  local_track = CTPPSLocalTrackLite(
    detid,
    ( kin_tr[0] + CLHEP::RandGauss::shoot( rnd_ ) * rp_resolution_x )*1.0e3, // in m
    rp_resolution_x * 1.0e3,
    ( kin_tr[2] + CLHEP::RandGauss::shoot( rnd_ ) * rp_resolution_y )*1.0e3, // in m
    rp_resolution_y * 1.0e3
  );

  return true;
}

//template<>
bool
CTPPSFastProtonSimulation::produceHit( const CLHEP::Hep3Vector& coord_local, TotemRPRecHit& rechit ) const
{
  double u = coord_local.x();
  double v = coord_local.y();

  // is it within detector?
  if ( !RPTopology::IsHit( u, v, insensitiveMargin_ ) ) return false;

  // round the measurement
  if ( roundToPitch_ ) {
    double m = stripZeroPosition_ - v;
    int strip = floor( m/pitch_ + 0.5 );
    v = stripZeroPosition_ - pitch_ * strip;
  }

  const double sigma = pitch_ / sqrt( 12. );

  rechit = TotemRPRecHit( v, sigma );
  return true;
}

//template<>


void
CTPPSFastProtonSimulation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSFastProtonSimulation );

