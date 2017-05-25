#include "SimRomanPot/CTPPSOpticsParameterisation/interface/ProtonReconstructionAlgorithm.h"

ProtonReconstructionAlgorithm::ProtonReconstructionAlgorithm( const edm::ParameterSet& beam_conditions, const std::string& optics_file ) :
  beamConditions_( beam_conditions )
{
  // load optics approximation
  TFile *f_in_optics = TFile::Open( optics_file.c_str() );
  if (f_in_optics == NULL) {
    throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't open file " << optics_file.c_str();
  }

  // build RP id, optics object name association
  std::map<unsigned int, std::string> nameMap = {
    { 2, "ip5_to_station_150_h_1_lhcb2" },
    { 3, "ip5_to_station_150_h_2_lhcb2" },
    { 102, "ip5_to_station_150_h_1_lhcb1" },
    { 103, "ip5_to_station_150_h_2_lhcb1" }
  };

  // build optics data for each object
  for (const auto &it : nameMap) {
    const unsigned int& rpId = it.first;
    const TotemRPDetId pot_id( rpId*10 );

    const std::string &ofName = it.second;

    LHCOpticsApproximator *of_orig = (LHCOpticsApproximator *) f_in_optics->Get(ofName.c_str());

    if (of_orig == NULL) {
      throw cms::Exception("ProtonReconstructionAlgorithm") << "Can't load object " << ofName;
    }

    RPOpticsData rpod;
    rpod.optics = new LHCOpticsApproximator(* of_orig);

    // build auxiliary optical functions
    double crossing_angle = 0.;
    double vtx0_y = 0.;

    // determine LHC sector from RP id
    if ( pot_id.arm()==0 ) {
      crossing_angle = beamConditions_.getParameter<double>( "halfCrossingAngleSector45" );
      vtx0_y = beamConditions_.getParameter<double>( "yOffsetSector45" );
    }

    if ( pot_id.arm()==1 ) {
      crossing_angle = beamConditions_.getParameter<double>( "halfCrossingAngleSector56" );
      vtx0_y = beamConditions_.getParameter<double>( "yOffsetSector56" );
    }

    const bool check_appertures = false;
    const bool invert_beam_coord_sytems = true;

    TGraph *g_xi_vs_x = new TGraph();
    TGraph *g_y0_vs_xi = new TGraph();
    TGraph *g_v_y_vs_xi = new TGraph();
    TGraph *g_L_y_vs_xi = new TGraph();

    for (double xi = 0.; xi <= 0.201; xi += 0.005) {
      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
      double kin_out_xi[5];
      rpod.optics->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_sytems);

      // input: xi and vtx_y
      const double vtx_y = 10E-6;	// m
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5];
      rpod.optics->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_sytems);

      // input: xi and th_y
      const double th_y = 20E-6;	// rad
      double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
      double kin_out_xi_th_y[5];
      rpod.optics->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_sytems);

      // fill graphs
      int idx = g_xi_vs_x->GetN();
      g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
      g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);
      g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
      g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);
    }

    rpod.s_xi_vs_x = new TSpline3("", g_xi_vs_x->GetX(), g_xi_vs_x->GetY(), g_xi_vs_x->GetN());
    delete g_xi_vs_x;

    rpod.s_y0_vs_xi = new TSpline3("", g_y0_vs_xi->GetX(), g_y0_vs_xi->GetY(), g_y0_vs_xi->GetN());
    delete g_y0_vs_xi;

    rpod.s_v_y_vs_xi = new TSpline3("", g_v_y_vs_xi->GetX(), g_v_y_vs_xi->GetY(), g_v_y_vs_xi->GetN());
    delete g_v_y_vs_xi;

    rpod.s_L_y_vs_xi = new TSpline3("", g_L_y_vs_xi->GetX(), g_L_y_vs_xi->GetY(), g_L_y_vs_xi->GetN());
    delete g_L_y_vs_xi;

   // insert optics data
    m_rp_optics[rpId] = rpod;
  }

  // initialise fitter
  chiSquareCalculator = new ChiSquareCalculator( beamConditions_ );

  fitter_ = std::make_unique<ROOT::Fit::Fitter>();
  double pStart[] = {0, 0, 0, 0};
  fitter_->SetFCN(4, *chiSquareCalculator, pStart, 0, true);//FIXME

  // clean up
  delete f_in_optics;
}

ProtonReconstructionAlgorithm::~ProtonReconstructionAlgorithm()
{
  for (auto &it : m_rp_optics) {
    delete it.second.optics;
    delete it.second.s_xi_vs_x;
    delete it.second.s_y0_vs_xi;
    delete it.second.s_v_y_vs_xi;
    delete it.second.s_L_y_vs_xi;
  }
  if (chiSquareCalculator) delete chiSquareCalculator;
}

//----------------------------------------------------------------------------------------------------

CTPPSSimProtonTrack
ProtonReconstructionAlgorithm::Reconstruct( const std::vector<CTPPSSimHit>& hits ) const
{
  CTPPSSimProtonTrack pd;

  if ( hits.size()<2 ) return pd;

  // check optics is available for all tracks
  for ( const auto& hit : hits ) {
    auto oit = m_rp_optics.find( hit.potId().rawId()/10 ); //FIXME
    if (oit == m_rp_optics.end()) {
      edm::LogProblem("ProtonReconstructionAlgorithm") << "Reconstruct > optics not available for RP " << hit.potId();
      return pd;
    }
  }

  // rough estimate of xi (mean of all xi's)
  double S_xi0 = 0., S_1 = 0.;

  for ( const auto& hit : hits ) {
    const unsigned short rp_id = hit.potId().rawId()/10; //FIXME
    auto oit = m_rp_optics.find( rp_id );
    double xi = oit->second.s_xi_vs_x->Eval( hit.getX0() );
    S_1 += 1.;
    S_xi0 += xi;
  }
  const double xi_0 = S_xi0 / S_1;

  // rough estimate of th_y and vtx_y
  double y[2], v_y[2], L_y[2];
  unsigned int y_idx = 0;
  for ( const auto& hit : hits ) {
    if ( y_idx>1 ) break;
    const unsigned short rp_id = hit.potId().rawId()/10; //FIXME
    auto oit = m_rp_optics.find( rp_id );

    y[y_idx] = hit.getY0()-oit->second.s_y0_vs_xi->Eval( xi_0 );
    v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval( xi_0 );
    L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval( xi_0 );

    y_idx++;
  }

  const double det = v_y[0] * L_y[1] - L_y[0] * v_y[1];
  const double vtx_y_0 = (L_y[1] * y[0] - L_y[0] * y[1]) / det;
  const double th_y_0 = (v_y[0] * y[1] - v_y[1] * y[0]) / det;
  //printf("    vtx_y_0 = %.3f mm\n", vtx_y_0 * 1E3);
  //printf("    th_y_0 = %.1f urad\n", th_y_0 * 1E6);

  // minimisation
  fitter_->Config().ParSettings(0).Set("xi", xi_0, 0.005);
  fitter_->Config().ParSettings(1).Set("th_x", 0., 20E-6);
  fitter_->Config().ParSettings(2).Set("th_y", th_y_0, 1E-6);
  fitter_->Config().ParSettings(3).Set("vtx_y", vtx_y_0, 1E-6);

  // TODO: this breaks the const-ness ??
  chiSquareCalculator->tracks = &hits;
  chiSquareCalculator->m_rp_optics = &m_rp_optics;

  fitter_->FitFCN();

  // extract proton parameters
  const ROOT::Fit::FitResult &result = fitter_->Result();
  const double *fitParameters = result.GetParams();

  pd.setValid( true );
  pd.setVertex( Local3DPoint( 0., fitParameters[3], 0. ) );
  pd.setDirection( Local3DVector( fitParameters[1], fitParameters[2], 0. ) );
  pd.setXi( fitParameters[0] );

  return pd;
}

//----------------------------------------------------------------------------------------------------

double
ProtonReconstructionAlgorithm::ChiSquareCalculator::operator() ( const double* parameters ) const
{
  // extract proton parameters
  const double &xi = parameters[0];
  const double &th_x = parameters[1];
  const double &th_y = parameters[2];
  const double vtx_x = 0;
  const double &vtx_y = parameters[3];

  // calculate chi^2
  double S2 = 0.;

  for (auto &it : *tracks) {
    double crossing_angle = 0.;
    double vtx0_y = 0.;
    const TotemRPDetId detid( it.potId() );

    // determine LHC sector from RP id
    if ( detid.arm()==0 ) {
      crossing_angle = beamConditions_.getParameter<double>( "halfCrossingAngleSector45" );
      vtx0_y = beamConditions_.getParameter<double>( "yOffsetSector45" );
    }

    if ( detid.arm()==1 ) {
      crossing_angle = beamConditions_.getParameter<double>( "halfCrossingAngleSector56" );
      vtx0_y = beamConditions_.getParameter<double>( "yOffsetSector56" );
    }

    // transport proton to the RP
    const unsigned int& rpId = detid.rawId()/10; //FIXME
    auto oit = m_rp_optics->find(rpId);

    const bool check_appertures = false;
    const bool invert_beam_coord_sytems = true;

    double kin_in[5] = { vtx_x,	(th_x + crossing_angle) * (1. - xi), vtx0_y + vtx_y, th_y * (1. - xi), -xi };
    double kin_out[5];
    oit->second.optics->Transport(kin_in, kin_out, check_appertures, invert_beam_coord_sytems);

    const double &x = kin_out[0];
    const double &y = kin_out[2];

    // calculate chi^2 constributions
    const double x_diff_norm = ( x-it.getX0() ) / it.getX0Sigma();
    const double y_diff_norm = ( y-it.getY0() ) / it.getY0Sigma();

    // increase chi^2
    S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
  }

  //printf("xi=%.3E, th_x=%.3E, th_y=%.3E, vtx_y=%.3E | S2 = %.3E\n", xi, th_x, th_y, vtx_y, S2);

  return S2;
}
