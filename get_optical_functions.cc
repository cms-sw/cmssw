#include "beam_conditions.h"

#include "SimG4CMS/TotemRPProtTranspPar/interface/LHCOpticsApproximator.h"

#include "TGraph.h"

#include <vector>
#include <string>

using namespace std;

//----------------------------------------------------------------------------------------------------

int cl_error = 0;

//----------------------------------------------------------------------------------------------------

bool TestBoolParameter(int argc, const char **argv, int &argi, const char *tag, bool &param1, bool &param2)
{
  if (strcmp(argv[argi], tag) == 0)
  {
    if (argi < argc - 1)
    {
      argi++;
      param1 = param2 = atoi(argv[argi]);
    } else {
      printf("ERROR: option '%s' requires an argument.\n", tag);
      cl_error = 1;
    }

    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------------------

bool TestBoolParameter(int argc, const char **argv, int &argi, const char *tag, bool &param)
{
  bool fake;
  return TestBoolParameter(argc, argv, argi, tag, param, fake);
}

//----------------------------------------------------------------------------------------------------

bool TestUIntParameter(int argc, const char **argv, int &argi, const char *tag, unsigned int &param)
{
  if (strcmp(argv[argi], tag) == 0)
  {
    if (argi < argc - 1)
    {
      argi++;
      param = (int) atof(argv[argi]);
    } else {
      printf("ERROR: option '%s' requires an argument.\n", tag);
      cl_error = 1;
    }

    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------------------

bool TestStringParameter(int argc, const char **argv, int &argi, const char *tag, string &param)
{
  if (strcmp(argv[argi], tag) == 0)
  {
    if (argi < argc - 1)
    {
      argi++;
      param = argv[argi];
    } else {
      printf("ERROR: option '%s' requires an argument.\n", tag);
      cl_error = 1;
    }

    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------------------

void PrintUsage()
{
  printf("USAGE: test_reconstruction [option] [option] ...\n");
  printf("OPTIONS:\n");
  printf("    -h, --help      print help and exit\n");
  
  printf("    -optics-file    specify optics file to use\n");

  printf("    -output         specify output file\n");
}

//----------------------------------------------------------------------------------------------------

int main(int argc, const char **argv)
{
  // defaults
  string file_optics = "parametrisation/version3/parametrization_6500GeV_0p4_185_reco.root";

  string file_output = "get_optical_functions.root";

  // parse command line
  for (int argi = 1; (argi < argc) && (cl_error == 0); ++argi)
  {
    if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0)
    {
      cl_error = 1;
      continue;
    }

    if (TestStringParameter(argc, argv, argi, "-optics-file", file_optics)) continue;

    if (TestStringParameter(argc, argv, argi, "-output", file_output)) continue;

    printf("ERROR: unknown option '%s'.\n", argv[argi]);
    cl_error = 1;
  }

  if (cl_error)
  {
    PrintUsage();
    return 1;
  }

  // print settings
  printf(">> settings\n");
  printf("    file_optics = %s\n", file_optics.c_str());
  printf("    file_output = %s\n", file_output.c_str());

  printf("\n");
  beamConditions.Print();

  // load optics
  TFile *f_in_optics = TFile::Open(file_optics.c_str());

  map<unsigned int, LHCOpticsApproximator*> optics;
  optics[2] = (LHCOpticsApproximator *) f_in_optics->Get("ip5_to_station_150_h_1_lhcb2");
  optics[3] = (LHCOpticsApproximator *) f_in_optics->Get("ip5_to_station_150_h_2_lhcb2");
  optics[102] = (LHCOpticsApproximator *) f_in_optics->Get("ip5_to_station_150_h_1_lhcb1");
  optics[103] = (LHCOpticsApproximator *) f_in_optics->Get("ip5_to_station_150_h_2_lhcb1");

  // prepare output
  TFile *f_out = TFile::Open(file_output.c_str(), "recreate");

  // sample functions for all RPs
  for (const auto oit : optics)
  {
    const unsigned int &rpId = oit.first;
    const LHCOpticsApproximator *optApp = oit.second;

    // determine LHC sector from RP id
    LHCSector sector = unknownSector;
    if ((rpId / 100) == 0)
      sector = sector45;
    if ((rpId / 100) == 1)
      sector = sector56;

    // determine crossing angle
    double crossing_angle = 0.;
    if (sector == sector45)
      crossing_angle = beamConditions.half_crossing_angle_45;
    if (sector == sector56)
      crossing_angle = beamConditions.half_crossing_angle_56;

    // book graphs
    char buf[100];   

    sprintf(buf, "RP%u", rpId);
    gDirectory = f_out->mkdir(buf);

    TGraph *g_x0_vs_xi = new TGraph();
    TGraph *g_y0_vs_xi = new TGraph();

    TGraph *g_v_x_vs_xi = new TGraph();
    TGraph *g_L_x_vs_xi = new TGraph();

    TGraph *g_v_y_vs_xi = new TGraph();
    TGraph *g_L_y_vs_xi = new TGraph();

    TGraph *g_xi_vs_x = new TGraph();

    // sample curves
    for (double xi = 0.; xi <= 0.201; xi += 0.001)
    {
      const bool check_appertures = false;
      const bool invert_beam_coord_sytems = true;

      // input: only xi
      double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y, 0., -xi };
      double kin_out_xi[5];
        optApp->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_sytems);
  
      // input: xi and vtx_x
      const double vtx_x = 10E-6;  // m
      double kin_in_xi_vtx_x[5] = { vtx_x, crossing_angle * (1. - xi), beamConditions.vtx0_y, 0., -xi };
      double kin_out_xi_vtx_x[5];
        optApp->Transport(kin_in_xi_vtx_x, kin_out_xi_vtx_x, check_appertures, invert_beam_coord_sytems);
  
      // input: xi and th_x
      const double th_x = 20E-6;  // rad
      double kin_in_xi_th_x[5] = { 0., (crossing_angle + th_x) * (1. - xi), beamConditions.vtx0_y, 0., -xi };
      double kin_out_xi_th_x[5];
        optApp->Transport(kin_in_xi_th_x, kin_out_xi_th_x, check_appertures, invert_beam_coord_sytems);
  
      // input: xi and vtx_y
      const double vtx_y = 10E-6;  // m
      double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y + vtx_y, 0., -xi };
      double kin_out_xi_vtx_y[5];
        optApp->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_sytems);
  
      // input: xi and th_y
      const double th_y = 20E-6;  // rad
      double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y, th_y * (1. - xi), -xi };
      double kin_out_xi_th_y[5];
        optApp->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_sytems);
  
      // fill graphs
      int idx = g_xi_vs_x->GetN();
      g_x0_vs_xi->SetPoint(idx, xi, kin_out_xi[0]);
      g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);

      g_v_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_x[0] - kin_out_xi[0]) / vtx_x);
      g_L_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_x[0] - kin_out_xi[0]) / th_x);

      g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
      g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);

      g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
    }

    // write graphs
    g_x0_vs_xi->Write("g_x0_vs_xi");
    g_y0_vs_xi->Write("g_y0_vs_xi");

    g_v_x_vs_xi->Write("g_v_x_vs_xi");
    g_L_x_vs_xi->Write("g_L_x_vs_xi");

    g_v_y_vs_xi->Write("g_v_y_vs_xi");
    g_L_y_vs_xi->Write("g_L_y_vs_xi");

    g_xi_vs_x->Write("g_xi_vs_x");
  }

  // clean up
  delete f_out;

  return 0;
}
