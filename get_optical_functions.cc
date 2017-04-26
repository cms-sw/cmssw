#include "beam_conditions.h"
#include "command_line_tools.h"

#include "LHCOpticsApproximator.h"

#include "TGraph.h"

#include <vector>
#include <string>

using namespace std;


//----------------------------------------------------------------------------------------------------

void PrintUsage()
{
	printf("USAGE: test_reconstruction [option] [option] ...\n");
	printf("OPTIONS:\n");
	printf("		-h, --help			print help and exit\n");
	
	printf("		-vtx-y-0				overwrite the value of vtx_y_0\n");

	printf("		-output				 specify output file\n");
}

//----------------------------------------------------------------------------------------------------

int main(int argc, const char **argv)
{
	// defaults
	string file_output = "get_optical_functions.root";

	// parse command line
	for (int argi = 1; (argi < argc) && (cl_error == 0); ++argi)
	{
		if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0)
		{
			cl_error = 1;
			continue;
		}

		if (TestStringParameter(argc, argv, argi, "-output", file_output)) continue;
		
		if (TestDoubleParameter(argc, argv, argi, "-vtx-y-0", beamConditions.vtx0_y_45, beamConditions.vtx0_y_56)) continue;

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
	printf("    file_output = %s\n", file_output.c_str());

	printf("\n");
	beamConditions.Print();

	// load optics
	map<unsigned int, LHCOpticsApproximator*> optics;

	TFile *f_in_optics_beam1 = TFile::Open("parametrisations/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root");
	optics[102] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_1_lhcb1");
	optics[103] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_2_lhcb1");

	TFile *f_in_optics_beam2 = TFile::Open("parametrisations/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root");
	optics[2] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_1_lhcb2");
	optics[3] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_2_lhcb2");

	// prepare output
	TFile *f_out = TFile::Open(file_output.c_str(), "recreate");

	// sample functions for all RPs
	for (const auto oit : optics)
	{
		const unsigned int &rpId = oit.first;
		LHCOpticsApproximator *optApp = oit.second;

		// determine LHC sector from RP id
		LHCSector sector = unknownSector;
		if ((rpId / 100) == 0)
			sector = sector45;
		if ((rpId / 100) == 1)
			sector = sector56;

		// determine crossing angle, vertex offset
		double crossing_angle = 0.;
		double vtx0_y = 0.;

		if (sector == sector45)
		{
			crossing_angle = beamConditions.half_crossing_angle_45;
			vtx0_y = beamConditions.vtx0_y_45;
		}

		if (sector == sector56)
		{
			crossing_angle = beamConditions.half_crossing_angle_56;
			vtx0_y = beamConditions.vtx0_y_56;
		}

		// book graphs
		char buf[100];	 

		sprintf(buf, "RP%u", rpId);
		gDirectory = f_out->mkdir(buf);

		TGraph *g_x0_vs_xi = new TGraph();
		TGraph *g_y0_vs_xi = new TGraph();
		TGraph *g_y0_vs_x0 = new TGraph();
		TGraph *g_y0_vs_x0so = new TGraph();
		TGraph *g_y0so_vs_x0so = new TGraph();

		TGraph *g_v_x_vs_xi = new TGraph();
		TGraph *g_L_x_vs_xi = new TGraph();

		TGraph *g_v_y_vs_xi = new TGraph();
		TGraph *g_L_y_vs_xi = new TGraph();

		TGraph *g_xi_vs_x = new TGraph();
		TGraph *g_xi_vs_xso = new TGraph();
		 
		const bool check_appertures = false;
		const bool invert_beam_coord_systems = true;

		// input: all zero
		double kin_in_zero[5] = { 0., crossing_angle, vtx0_y, 0., 0. };
		double kin_out_zero[5] = { 0., 0., 0., 0., 0. };
		optApp->Transport(kin_in_zero, kin_out_zero, check_appertures, invert_beam_coord_systems);

		// sample curves
		for (double xi = 0.; xi <= 0.151; xi += 0.001)
		{
			// input: only xi
			double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), vtx0_y, 0., -xi };
			double kin_out_xi[5] = { 0., 0., 0., 0., 0. };
			optApp->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_systems);
	
			// input: xi and vtx_x
			const double vtx_x = 10E-6;	// m
			double kin_in_xi_vtx_x[5] = { vtx_x, crossing_angle * (1. - xi), vtx0_y, 0., -xi };
			double kin_out_xi_vtx_x[5] = { 0., 0., 0., 0., 0. };
			optApp->Transport(kin_in_xi_vtx_x, kin_out_xi_vtx_x, check_appertures, invert_beam_coord_systems);
	
			// input: xi and th_x
			const double th_x = 20E-6;	// rad
			double kin_in_xi_th_x[5] = { 0., (crossing_angle + th_x) * (1. - xi), vtx0_y, 0., -xi };
			double kin_out_xi_th_x[5] = { 0., 0., 0., 0., 0. };
			optApp->Transport(kin_in_xi_th_x, kin_out_xi_th_x, check_appertures, invert_beam_coord_systems);
	
			// input: xi and vtx_y
			const double vtx_y = 10E-6;	// m
			double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y + vtx_y, 0., -xi };
			double kin_out_xi_vtx_y[5] = { 0., 0., 0., 0., 0. };
			optApp->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_systems);
	
			// input: xi and th_y
			const double th_y = 20E-6;	// rad
			double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), vtx0_y, th_y * (1. - xi), -xi };
			double kin_out_xi_th_y[5] = { 0., 0., 0., 0., 0. };
			optApp->Transport(kin_in_xi_th_y, kin_out_xi_th_y, check_appertures, invert_beam_coord_systems);
	
			// fill graphs
			int idx = g_xi_vs_x->GetN();
			g_x0_vs_xi->SetPoint(idx, xi, kin_out_xi[0]);
			g_y0_vs_xi->SetPoint(idx, xi, kin_out_xi[2]);
			g_y0_vs_x0->SetPoint(idx, kin_out_xi[0], kin_out_xi[2]);
			g_y0_vs_x0so->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2]);
			g_y0so_vs_x0so->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], kin_out_xi[2] - kin_out_zero[2]);

			g_v_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_x[0] - kin_out_xi[0]) / vtx_x);
			g_L_x_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_x[0] - kin_out_xi[0]) / th_x);

			g_v_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_vtx_y[2] - kin_out_xi[2]) / vtx_y);
			g_L_y_vs_xi->SetPoint(idx, xi, (kin_out_xi_th_y[2] - kin_out_xi[2]) / th_y);

			g_xi_vs_x->SetPoint(idx, kin_out_xi[0], xi);
			g_xi_vs_xso->SetPoint(idx, kin_out_xi[0] - kin_out_zero[0], xi);
		}

		// write graphs
		g_x0_vs_xi->Write("g_x0_vs_xi");
		g_y0_vs_xi->Write("g_y0_vs_xi");
		g_y0_vs_x0->Write("g_y0_vs_x0");
		g_y0_vs_x0so->Write("g_y0_vs_x0so");
		g_y0so_vs_x0so->Write("g_y0so_vs_x0so");

		g_v_x_vs_xi->Write("g_v_x_vs_xi");
		g_L_x_vs_xi->Write("g_L_x_vs_xi");

		g_v_y_vs_xi->Write("g_v_y_vs_xi");
		g_L_y_vs_xi->Write("g_L_y_vs_xi");

		g_xi_vs_x->Write("g_xi_vs_x");
		g_xi_vs_xso->Write("g_xi_vs_xso");
	}

	// clean up
	delete f_out;

	return 0;
}
