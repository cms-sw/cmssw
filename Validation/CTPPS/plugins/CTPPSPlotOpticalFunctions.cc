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

class CTPPSPlotOpticalFunctions : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
  public:
    explicit CTPPSPlotOpticalFunctions(const edm::ParameterSet&);
    ~CTPPSPlotOpticalFunctions();

  private: 
    virtual void beginJob() override;
    virtual void analyze( const edm::Event&, const edm::EventSetup& ) {}

    edm::FileInPath opticsFile_;
    std::vector<std::string> opticsObjects_;

    double vtx0_y_45_, vtx0_y_56_;
    double half_crossing_angle_45_, half_crossing_angle_56_;

    // book graphs
    std::map<std::string,TGraph*> g_x0_vs_xi, g_y0_vs_xi, g_y0_vs_x0, g_y0_vs_x0so, g_y0so_vs_x0so;
    std::map<std::string,TGraph*> g_v_x_vs_xi, g_L_x_vs_xi, g_v_y_vs_xi, g_L_y_vs_xi;
    std::map<std::string,TGraph*> g_xi_vs_x, g_xi_vs_xso;
     
};

//----------------------------------------------------------------------------------------------------

CTPPSPlotOpticalFunctions::CTPPSPlotOpticalFunctions( const edm::ParameterSet& iConfig ) :
  opticsFile_            ( iConfig.getParameter<edm::FileInPath>( "opticsFile" ) ),
  opticsObjects_         ( iConfig.getParameter< std::vector<std::string> >( "opticsObjects" ) ),
  vtx0_y_45_             ( iConfig.getParameter<double>( "vtx0_y_45" ) ),
  vtx0_y_56_             ( iConfig.getParameter<double>( "vtx0_y_56" ) ),
  half_crossing_angle_45_( iConfig.getParameter<double>( "half_crossing_angle_45" ) ),
  half_crossing_angle_56_( iConfig.getParameter<double>( "half_crossing_angle_56" ) )
{
  usesResource( "TFileService" );

  // prepare output
  edm::Service<TFileService> fs;

  // book graphs
  for ( const auto& objName : opticsObjects_ ) {
    // make output directory
    TFileDirectory dir = fs->mkdir( objName.c_str() );
    g_x0_vs_xi[objName] = dir.make<TGraph>();
    g_y0_vs_xi[objName] = dir.make<TGraph>();
    g_y0_vs_x0[objName] = dir.make<TGraph>();
    g_y0_vs_x0so[objName] = dir.make<TGraph>();
    g_y0so_vs_x0so[objName] = dir.make<TGraph>();

    g_v_x_vs_xi[objName] = dir.make<TGraph>();
    g_L_x_vs_xi[objName] = dir.make<TGraph>();

    g_v_y_vs_xi[objName] = dir.make<TGraph>();
    g_L_y_vs_xi[objName] = dir.make<TGraph>();

    g_xi_vs_x[objName] = dir.make<TGraph>();
    g_xi_vs_xso[objName] = dir.make<TGraph>();
    /*g_x0_vs_xi->Write("g_x0_vs_xi");
    g_y0_vs_xi->Write("g_y0_vs_xi");
    g_y0_vs_x0->Write("g_y0_vs_x0");
    g_y0_vs_x0so->Write("g_y0_vs_x0so");
    g_y0so_vs_x0so->Write("g_y0so_vs_x0so");

    g_v_x_vs_xi->Write("g_v_x_vs_xi");
    g_L_x_vs_xi->Write("g_L_x_vs_xi");

    g_v_y_vs_xi->Write("g_v_y_vs_xi");
    g_L_y_vs_xi->Write("g_L_y_vs_xi");

    g_xi_vs_x->Write("g_xi_vs_x");
    g_xi_vs_xso->Write("g_xi_vs_xso");*/
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSPlotOpticalFunctions::~CTPPSPlotOpticalFunctions()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSPlotOpticalFunctions::beginJob()
{
  printf(">> CTPPSPlotOpticalFunctions::beginJob\n");

  // open input file
  auto f_in = std::make_unique<TFile>( opticsFile_.fullPath().c_str() );
  if ( !f_in )
    throw cms::Exception("CTPPSPlotOpticalFunctions") << "Cannot open file '" << opticsFile_ << "'.";

  // go through all optics objects
  for ( const auto& objName : opticsObjects_ ) {
    const auto optApp = dynamic_cast<LHCOpticsApproximator*>( f_in->Get( objName.c_str() ) );
    if (!optApp)
      throw cms::Exception("CTPPSPlotOpticalFunctions") << "Cannot load object '" << objName << "'.";

    printf("* %s --> %s\n", objName.c_str(), optApp->GetName());

    // determine crossing angle, vertex offset
    double crossing_angle = 0.;
    double vtx0_y = 0.;

    if (optApp->GetBeamType() == LHCOpticsApproximator::lhcb2) {
      crossing_angle = half_crossing_angle_45_;
      vtx0_y = vtx0_y_45_;
    }

    if (optApp->GetBeamType() == LHCOpticsApproximator::lhcb1) {
      crossing_angle = half_crossing_angle_56_;
      vtx0_y = vtx0_y_56_;
    }

    const bool check_apertures = false;
    const bool invert_beam_coord_systems = true;

    // input: all zero
    double kin_in_zero[5] = { 0., crossing_angle, vtx0_y, 0., 0. };
    double kin_out_zero[5] = { 0., 0., 0., 0., 0. };
    optApp->Transport(kin_in_zero, kin_out_zero, check_apertures, invert_beam_coord_systems);

    // sample curves
    for (double xi = 0.; xi <= 0.151; xi += 0.001) {
      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi, kin_out_xi, check_apertures, invert_beam_coord_systems);
  
      // input: xi and vtx_x
      const double vtx_x = 10E-6;  // m
      double kin_in_xi_vtx_x[5] = { vtx_x, crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi_vtx_x[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_vtx_x, kin_out_xi_vtx_x, check_apertures, invert_beam_coord_systems);
  
      // input: xi and th_x
      const double th_x = 20E-6;  // rad
      double kin_in_xi_th_x[5] = { 0., (crossing_angle + th_x) * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi_th_x[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_th_x, kin_out_xi_th_x, check_apertures, invert_beam_coord_systems);
  
      // input: xi and vtx_y
      const double vtx_y = 10E-6;  // m
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_apertures, invert_beam_coord_systems);
  
      // input: xi and th_y
      const double th_y = 20E-6;  // rad
      double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
      double kin_out_xi_th_y[5] = { 0., 0., 0., 0., 0. };
      optApp->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_apertures, invert_beam_coord_systems);
  
      // fill graphs
      int idx = g_xi_vs_x[objName]->GetN();
      g_x0_vs_xi[objName]->SetPoint(idx, xi, kin_out_xi[0]);
      g_y0_vs_xi[objName]->SetPoint(idx, xi, kin_out_xi[2]);
      g_y0_vs_x0[objName]->SetPoint(idx, kin_out_xi[0], kin_out_xi[2]);
      g_y0_vs_x0so[objName]->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2]);
      g_y0so_vs_x0so[objName]->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2] - kin_out_zero[2]);

      g_v_x_vs_xi[objName]->SetPoint(idx, xi, (kin_out_xi_vtx_x[0] - kin_out_xi[0]) / vtx_x);
      g_L_x_vs_xi[objName]->SetPoint(idx, xi, (kin_out_xi_th_x[0] - kin_out_xi[0]) / th_x);

      g_v_y_vs_xi[objName]->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
      g_L_y_vs_xi[objName]->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);

      g_xi_vs_x[objName]->SetPoint(idx, kin_out_xi[0], xi);
      g_xi_vs_xso[objName]->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], xi);
    }

  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSPlotOpticalFunctions );

