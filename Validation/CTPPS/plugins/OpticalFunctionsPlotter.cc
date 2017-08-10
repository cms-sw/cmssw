/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/CTPPSOpticsObjects/interface/LHCOpticsApproximator.h"

#include <map>

#include "TGraph.h"

//----------------------------------------------------------------------------------------------------

class OpticalFunctionsPlotter : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
  public:
    explicit OpticalFunctionsPlotter(const edm::ParameterSet&);
    ~OpticalFunctionsPlotter();

  private: 
    virtual void beginJob() override;
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) {}

    edm::FileInPath opticsFile_;
    std::vector<std::string> opticsObjects_;
    edm::ParameterSet beamConditions_;

    double vertex_size_, beam_divergence_;
    double vtx0_y_45_, vtx0_y_56_;
    double half_crossing_angle_45_, half_crossing_angle_56_;

    double minXi_, maxXi_, xiStep_;

    // book graphs
    std::map<std::string,TGraph*> g_x0_vs_xi, g_y0_vs_xi, g_y0_vs_x0, g_y0_vs_x0so, g_y0so_vs_x0so;
    std::map<std::string,TGraph*> g_v_x_vs_xi, g_L_x_vs_xi, g_v_y_vs_xi, g_L_y_vs_xi;
    std::map<std::string,TGraph*> g_xi_vs_x, g_xi_vs_xso;
     
};

//----------------------------------------------------------------------------------------------------

OpticalFunctionsPlotter::OpticalFunctionsPlotter( const edm::ParameterSet& iConfig ) :
  opticsFile_            ( iConfig.getParameter<edm::FileInPath>( "opticsFile" ) ),
  opticsObjects_         ( iConfig.getParameter< std::vector<std::string> >( "opticsObjects" ) ),
  beamConditions_        ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  vertex_size_           ( beamConditions_.getParameter<double>( "vertexSize" ) ), // in m
  beam_divergence_       ( beamConditions_.getParameter<double>( "beamDivergence" ) ), // in rad
  vtx0_y_45_             ( beamConditions_.getParameter<double>( "yOffsetSector45" ) ), // in m
  vtx0_y_56_             ( beamConditions_.getParameter<double>( "yOffsetSector56" ) ), // in m
  half_crossing_angle_45_( beamConditions_.getParameter<double>( "halfCrossingAngleSector45" ) ), // in rad
  half_crossing_angle_56_( beamConditions_.getParameter<double>( "halfCrossingAngleSector56" ) ), // in rad
  minXi_                 ( iConfig.getParameter<double>( "minXi" ) ),
  maxXi_                 ( iConfig.getParameter<double>( "maxXi" ) ),
  xiStep_                ( iConfig.getParameter<double>( "xiStep" ) )
{
  usesResource( "TFileService" );

  // prepare output
  edm::Service<TFileService> fs;

  // book graphs
  for ( const auto& objName : opticsObjects_ ) {
    // make output directory
    TFileDirectory dir = fs->mkdir( objName.c_str() );
    g_x0_vs_xi[objName] = dir.make<TGraph>();
    g_x0_vs_xi[objName]->SetName( "g_x0_vs_xi" );
    g_x0_vs_xi[objName]->SetTitle( ";#xi;x_{0}" );
    g_y0_vs_xi[objName] = dir.make<TGraph>();
    g_y0_vs_xi[objName]->SetName( "g_y0_vs_xi" );
    g_y0_vs_xi[objName]->SetTitle( ";#xi;y_{0}" );
    g_y0_vs_x0[objName] = dir.make<TGraph>();
    g_y0_vs_x0[objName]->SetName( "g_y0_vs_x0" );
    g_y0_vs_x0[objName]->SetTitle( ";x_{0};y_{0}" );
    g_y0_vs_x0so[objName] = dir.make<TGraph>();
    g_y0_vs_x0so[objName]->SetName( "g_y0_vs_x0so" );
    g_y0_vs_x0so[objName]->SetTitle( ";#hat{x}_{0};y_{0}" );
    g_y0so_vs_x0so[objName] = dir.make<TGraph>();
    g_y0so_vs_x0so[objName]->SetName( "g_y0so_vs_x0so" );
    g_y0so_vs_x0so[objName]->SetTitle( ";#hat{x}_{0};#hat{y}_{0}" );

    g_v_x_vs_xi[objName] = dir.make<TGraph>();
    g_v_x_vs_xi[objName]->SetName( "g_v_x_vs_xi" );
    g_v_x_vs_xi[objName]->SetTitle( ";#xi;v_{x}" );
    g_L_x_vs_xi[objName] = dir.make<TGraph>();
    g_L_x_vs_xi[objName]->SetName( "g_L_x_vs_xi" );
    g_L_x_vs_xi[objName]->SetTitle( ";#xi;L_{x}" );

    g_v_y_vs_xi[objName] = dir.make<TGraph>();
    g_v_y_vs_xi[objName]->SetName( "g_v_y_vs_xi" );
    g_v_y_vs_xi[objName]->SetTitle( ";#xi;v_{y}" );
    g_L_y_vs_xi[objName] = dir.make<TGraph>();
    g_L_y_vs_xi[objName]->SetName( "g_L_y_vs_xi" );
    g_L_y_vs_xi[objName]->SetTitle( ";#xi;L_{y}" );

    g_xi_vs_x[objName] = dir.make<TGraph>();
    g_xi_vs_x[objName]->SetName( "g_xi_vs_x" );
    g_xi_vs_x[objName]->SetTitle( ";x;#xi" );
    g_xi_vs_xso[objName] = dir.make<TGraph>();
    g_xi_vs_xso[objName]->SetName( "g_xi_vs_xso" );
    g_xi_vs_xso[objName]->SetTitle( ";#hat{x};#xi" );
  }
}

//----------------------------------------------------------------------------------------------------

OpticalFunctionsPlotter::~OpticalFunctionsPlotter()
{}

//----------------------------------------------------------------------------------------------------

void
OpticalFunctionsPlotter::beginJob()
{
  std::ostringstream oss;

  // open input file
  auto f_in = std::make_unique<TFile>( opticsFile_.fullPath().c_str() );
  if ( !f_in )
    throw cms::Exception("OpticalFunctionsPlotter") << "Cannot open file '" << opticsFile_ << "'.";

  // go through all optics objects
  for ( const auto& objName : opticsObjects_ ) {
    const auto optApp = dynamic_cast<LHCOpticsApproximator*>( f_in->Get( objName.c_str() ) );
    if (!optApp)
      throw cms::Exception("OpticalFunctionsPlotter") << "Cannot load object '" << objName << "'.";

    oss << "* " << objName << " --> " << optApp->GetName() << std::endl;

    // determine crossing angle, vertex offset
    double crossing_angle = 0.0;
    double vtx0_y = 0.0;

    if ( optApp->GetBeamType()==LHCOpticsApproximator::lhcb2 ) {
      crossing_angle = half_crossing_angle_45_;
      vtx0_y = vtx0_y_45_;
    }

    if ( optApp->GetBeamType()==LHCOpticsApproximator::lhcb1 ) {
      crossing_angle = half_crossing_angle_56_;
      vtx0_y = vtx0_y_56_;
    }

    const bool check_apertures = false;
    const bool invert_beam_coord_systems = true;

    // input: all zero
    std::array<double,5> kin_in_zero, kin_out_zero;
    kin_in_zero = { { 0.0, crossing_angle, vtx0_y, 0.0, 0.0 } };
    optApp->Transport( kin_in_zero.data(), kin_out_zero.data(), check_apertures, invert_beam_coord_systems );

    // sample curves
    for ( double xi=minXi_; xi<=maxXi_; xi+=xiStep_ ) {
      // input: only xi
      std::array<double,5> kin_in_xi = { { 0.0, crossing_angle * ( 1.-xi ), vtx0_y, 0.0, -xi } }, kin_out_xi;
      optApp->Transport( kin_in_xi.data(), kin_out_xi.data(), check_apertures, invert_beam_coord_systems );
  
      // input: xi and vtx_x (vertex size)
      std::array<double,5> kin_in_xi_vtx_x = { { vertex_size_, crossing_angle * ( 1.-xi ), vtx0_y, 0.0, -xi } }, kin_out_xi_vtx_x;
      optApp->Transport( kin_in_xi_vtx_x.data(), kin_out_xi_vtx_x.data(), check_apertures, invert_beam_coord_systems );
  
      // input: xi and th_x (beam divergence)
      std::array<double,5> kin_in_xi_th_x = { { 0.0, ( crossing_angle+beam_divergence_ ) * ( 1.-xi ), vtx0_y, 0.0, -xi } }, kin_out_xi_th_x;
      optApp->Transport( kin_in_xi_th_x.data(), kin_out_xi_th_x.data(), check_apertures, invert_beam_coord_systems );
  
      // input: xi and vtx_y (vertex size)
      std::array<double,5> kin_in_xi_vtx_y = { { 0.0, crossing_angle * ( 1.-xi ), vtx0_y + vertex_size_, 0.0, -xi } }, kin_out_xi_vtx_y;
      optApp->Transport( kin_in_xi_vtx_y.data(), kin_out_xi_vtx_y.data(), check_apertures, invert_beam_coord_systems );
  
      // input: xi and th_y (beam divergence)
      std::array<double,5> kin_in_xi_th_y = { { 0.0, crossing_angle * ( 1.-xi ), vtx0_y, beam_divergence_ * ( 1.-xi ), -xi } }, kin_out_xi_th_y;
      optApp->Transport( kin_in_xi_th_y.data(), kin_out_xi_th_y.data(), check_apertures, invert_beam_coord_systems );
  
      // fill graphs
      int idx = g_xi_vs_x[objName]->GetN();
      g_x0_vs_xi[objName]->SetPoint( idx, xi, kin_out_xi[0] );
      g_y0_vs_xi[objName]->SetPoint( idx, xi, kin_out_xi[2] );
      g_y0_vs_x0[objName]->SetPoint( idx, kin_out_xi[0], kin_out_xi[2] );
      g_y0_vs_x0so[objName]->SetPoint( idx, kin_out_xi[0]-kin_out_zero[0], kin_out_xi[2] );
      g_y0so_vs_x0so[objName]->SetPoint( idx, kin_out_xi[0]-kin_out_zero[0], kin_out_xi[2]-kin_out_zero[2] );

      g_v_x_vs_xi[objName]->SetPoint( idx, xi, ( kin_out_xi_vtx_x[0]-kin_out_xi[0] )/vertex_size_ );
      g_L_x_vs_xi[objName]->SetPoint( idx, xi, ( kin_out_xi_th_x[0]-kin_out_xi[0] )/beam_divergence_ );

      g_v_y_vs_xi[objName]->SetPoint( idx, xi, ( kin_out_xi_vtx_y[2]-kin_out_xi[2] )/vertex_size_ );
      g_L_y_vs_xi[objName]->SetPoint( idx, xi, ( kin_out_xi_th_y[2]-kin_out_xi[2] )/beam_divergence_ );

      g_xi_vs_x[objName]->SetPoint( idx, kin_out_xi[0], xi );
      g_xi_vs_xso[objName]->SetPoint( idx, kin_out_xi[0]-kin_out_zero[0], xi );
    }
  }
  edm::LogInfo("OpticalFunctionsPlotter::beginJob") << oss.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( OpticalFunctionsPlotter );

