#include "track_lite.h"
#include "beam_conditions.h"
#include "proton_reconstruction.h"

#include "LHCOpticsApproximator.h"

#include "TRandom3.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"

#include <vector>
#include <string>

using namespace std;

//----------------------------------------------------------------------------------------------------

bool debug = false;

//----------------------------------------------------------------------------------------------------

TGraphErrors* ProfileToRMSGraph(TProfile *p, const string &name = "")
{
	TGraphErrors *g = new TGraphErrors();
	g->SetName(name.c_str());

	for (int bi = 1; bi <= p->GetNbinsX(); ++bi)
	{
		double c = p->GetBinCenter(bi);

		double N = p->GetBinEntries(bi);
		double Sy = p->GetBinContent(bi) * N;
		double Syy = p->GetSumw2()->At(bi);

		double si_sq = Syy/N - Sy*Sy/N/N;
		double si = (si_sq >= 0.) ? sqrt(si_sq) : 0.;
		double si_unc_sq = si_sq / 2. / N;	// Gaussian approximation
		double si_unc = (si_unc_sq >= 0.) ? sqrt(si_unc_sq) : 0.;

		int idx = g->GetN();
		g->SetPoint(idx, c, si);
		g->SetPointError(idx, 0., si_unc);
	}

	return g;
}

//----------------------------------------------------------------------------------------------------

/// implemented according to LHCOpticsApproximator::Transport_m_GeV
/// xi is positive for diffractive protons, thus proton momentum p = (1 - xi) * p_nom
/// horizontal component of proton momentum: p_x = th_x * (1 - xi) * p_nom

void BuildTrackCollection(LHCSector sector, double vtx_x, double vtx_y, double th_x, double th_y, double xi,
			const map<unsigned int, LHCOpticsApproximator*> &optics, TrackDataCollection &tracks)
{
	// settings
	const bool check_appertures = true;
	const bool invert_beam_coord_sytems = true;

	// start with no tracks
	tracks.clear();

	// convert physics kinematics to the LHC reference frame
	vtx_y += beamConditions.vtx0_y;
	if (sector == sector45)
		th_x += beamConditions.half_crossing_angle_45;
	if (sector == sector56)
		th_x += beamConditions.half_crossing_angle_56;

	// transport proton to each RP
	for (const auto it : optics)
	{
		double kin_in[5];
		kin_in[0] = vtx_x;
		kin_in[1] = th_x * (1. - xi);
		kin_in[2] = vtx_y;
		kin_in[3] = th_y * (1. - xi);
		kin_in[4] = - xi;

		double kin_out[5];
		bool proton_trasported = it.second->Transport(kin_in, kin_out, check_appertures, invert_beam_coord_sytems);

		// stop if proton not transportable
		if (!proton_trasported)
			continue;

		// add track
		TrackData td;
		td.valid = true;
		td.x = kin_out[0];
		td.y = kin_out[2];
		td.x_unc = 12E-6;
		td.y_unc = 12E-6;

		tracks[it.first] = td;

		if (debug)
			printf("		RP %u: x = %.3f mm, y = %.3f mm\n", it.first, td.x*1E3, td.y*1E3);

	}
}

//----------------------------------------------------------------------------------------------------
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

bool TestStringParameter(int argc, const char **argv, int &argi, const char *tag, string &param1, string &param2)
{
	if (strcmp(argv[argi], tag) == 0)
	{
		if (argi < argc - 1)
		{
			argi++;
			param1 = param2 = argv[argi];
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
	string fake;
	return TestStringParameter(argc, argv, argi, tag, param, fake);
}

//----------------------------------------------------------------------------------------------------

void PrintUsage()
{
	printf("USAGE: test_reconstruction [option] [option] ...\n");
	printf("OPTIONS:\n");
	printf("    -h, --help           print help and exit\n");

	printf("    -sim-vtx             set on/off generation of vertex x and y\n");
	printf("    -sim-vtx-x           set on/off generation of vertex x\n");
	printf("    -sim-vtx-y           set on/off generation of vertex y\n");

	printf("    -sim-ang             set on/off generation of angles x and y\n");
	printf("    -sim-ang-x           set on/off generation of angles x\n");
	printf("    -sim-ang-y           set on/off generation of angles y\n");

	printf("    -sim-beam-div        set on/off generation of beam divergence x and y\n");
	printf("    -sim-xi              set on/off generation of xi\n");
	printf("    -sim-det-res         set on/off generation of detector resolution x and y\n");

	printf("    -events              set number of events to simulate\n");
	printf("    -seed                set random seed\n");

	printf("    -optics-file         specify optics file to use (for both beam 1 and 2)\n");
	printf("    -optics-file-beam1   specify optics file to use for beam 1\n");
	printf("    -optics-file-beam2   specify optics file to use for beam 2\n");

	printf("    -output              specify output file\n");
}

//----------------------------------------------------------------------------------------------------

int main(int argc, const char **argv)
{
	// defaults
	bool simulate_vertex_x = true;
	bool simulate_vertex_y = true;
	bool simulate_scattering_angles_x = true;
	bool simulate_scattering_angles_y = true;
	bool simulate_beam_divergence = false;
	bool simulate_xi = true;
	bool simulate_detector_resolution = false;

	unsigned int n_events = 1000;

	unsigned int seed = 1;

	string file_optics_beam1 = "parametrisations/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root";
	string file_optics_beam2 = "parametrisations/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root";

	string file_output = "test_reconstruction.root";

	double si_th_phys = 25E-6;	// physics scattering angle, rad
	double si_det = 12E-6;			// RP resolution, m
	double xi_min = 0.03;
	double xi_max = 0.17;

	// parse command line
	for (int argi = 1; (argi < argc) && (cl_error == 0); ++argi)
	{
		if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0)
		{
			cl_error = 1;
			continue;
		}

		if (TestBoolParameter(argc, argv, argi, "-sim-vtx", simulate_vertex_x, simulate_vertex_y)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-vtx-x", simulate_vertex_x)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-vtx-y", simulate_vertex_y)) continue;

		if (TestBoolParameter(argc, argv, argi, "-sim-ang", simulate_scattering_angles_x, simulate_scattering_angles_y)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-ang-x", simulate_scattering_angles_x)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-ang-y", simulate_scattering_angles_y)) continue;

		if (TestBoolParameter(argc, argv, argi, "-sim-beam-div", simulate_beam_divergence)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-xi", simulate_xi)) continue;
		if (TestBoolParameter(argc, argv, argi, "-sim-det-res", simulate_detector_resolution)) continue;

		if (TestUIntParameter(argc, argv, argi, "-events", n_events)) continue;
		if (TestUIntParameter(argc, argv, argi, "-seed", seed)) continue;

		if (TestStringParameter(argc, argv, argi, "-optics-file", file_optics_beam1, file_optics_beam2)) continue;
		if (TestStringParameter(argc, argv, argi, "-optics-file-beam1", file_optics_beam1)) continue;
		if (TestStringParameter(argc, argv, argi, "-optics-file-beam2", file_optics_beam2)) continue;

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
	printf("		simulate_vertex_x = %u\n", simulate_vertex_x);
	printf("		simulate_vertex_y = %u\n", simulate_vertex_y);
	printf("		simulate_scattering_angles_x = %u\n", simulate_scattering_angles_x);
	printf("		simulate_scattering_angles_y = %u\n", simulate_scattering_angles_y);
	printf("		simulate_beam_divergence = %u\n", simulate_beam_divergence);
	printf("		simulate_xi = %u\n", simulate_xi);
	printf("		simulate_detector_resolution = %u\n", simulate_detector_resolution);
	printf("		n_events = %u (%.1E)\n", n_events, double(n_events));
	printf("		seed = %u\n", seed);
	printf("		file_optics_beam1 = %s\n", file_optics_beam1.c_str());
	printf("		file_optics_beam2 = %s\n", file_optics_beam2.c_str());
	printf("		file_output = %s\n", file_output.c_str());

	printf("\n");
	beamConditions.Print();

	// load optics
	map<unsigned int, LHCOpticsApproximator*> optics_45, optics_56; // map: RP id --> optics

	TFile *f_in_optics_beam1 = TFile::Open(file_optics_beam1.c_str());
	optics_56[102] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_1_lhcb1");
	optics_56[103] = (LHCOpticsApproximator *) f_in_optics_beam1->Get("ip5_to_station_150_h_2_lhcb1");

	TFile *f_in_optics_beam2 = TFile::Open(file_optics_beam2.c_str());
	optics_45[2] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_1_lhcb2");
	optics_45[3] = (LHCOpticsApproximator *) f_in_optics_beam2->Get("ip5_to_station_150_h_2_lhcb2");

	// initialise proton reconstruction
	ProtonReconstruction protonReconstruction;
	if (protonReconstruction.Init(file_optics_beam1, file_optics_beam2) != 0)
		return 1;

	// prepare plots - hit distributions
	map<unsigned int, TH2D*> m_rp_h2_y_vs_x;
	for (unsigned int rpId : { 2, 3, 102, 103} )
	{
		m_rp_h2_y_vs_x[rpId] = new TH2D("", ";x;y", 300, 0., 30E-3, 200, -10E-3, +10E-3);
	}

	// prepare plots - histograms
	TH1D *h_de_vtx_x_45 = new TH1D("h_de_vtx_x_45", ";vtx_{x}^{reco,45} - vtx_{x}^{sim}", 100, -40E-6, +40E-6);
	TH1D *h_de_vtx_x_56 = new TH1D("h_de_vtx_x_56", ";vtx_{x}^{reco,56} - vtx_{x}^{sim}", 100, -40E-6, +40E-6);

	TH1D *h_de_vtx_y_45 = new TH1D("h_de_vtx_y_45", ";vtx_{y}^{reco,45} - vtx_{y}^{sim}", 100, -250E-6, +250E-6);
	TH1D *h_de_vtx_y_56 = new TH1D("h_de_vtx_y_56", ";vtx_{y}^{reco,56} - vtx_{y}^{sim}", 100, -250E-6, +250E-6);

	TH1D *h_de_th_x_45 = new TH1D("h_de_th_x_45", ";th_{x}^{reco,45} - th_{x}^{sim}", 100, -100E-6, +100E-6);
	TH1D *h_de_th_x_56 = new TH1D("h_de_th_x_56", ";th_{x}^{reco,56} - th_{x}^{sim}", 100, -100E-6, +100E-6);

	TH1D *h_de_th_y_45 = new TH1D("h_de_th_y_45", ";th_{y}^{reco,45} - th_{y}^{sim}", 100, -100E-6, +100E-6);
	TH1D *h_de_th_y_56 = new TH1D("h_de_th_y_56", ";th_{y}^{reco,56} - th_{y}^{sim}", 100, -100E-6, +100E-6);

	TH1D *h_de_xi_45 = new TH1D("h_de_xi_45", ";#xi^{reco,45} - #xi^{sim}", 100, -5E-3, +5E-3);
	TH1D *h_de_xi_56 = new TH1D("h_de_xi_56", ";#xi^{reco,56} - #xi^{sim}", 100, -5E-3, +5E-3);

	// prepare plots - 2D histograms
	TH2D *h2_de_vtx_x_vs_de_xi_45 = new TH2D("h2_de_vtx_x_vs_de_xi_45", ";#Delta#xi^{45};#Deltavtx_{x}^{45}", 50, -5E-3, +5E-3, 50, -40E-6, +40E-6);
	TH2D *h2_de_vtx_x_vs_de_xi_56 = new TH2D("h2_de_vtx_x_vs_de_xi_56", ";#Delta#xi^{56};#Deltavtx_{x}^{56}", 50, -5E-3, +5E-3, 50, -40E-6, +40E-6);

	TH2D *h2_de_vtx_y_vs_de_xi_45 = new TH2D("h2_de_vtx_y_vs_de_xi_45", ";#Delta#xi^{45};#Deltavtx_{y}^{45}", 50, -5E-3, +5E-3, 50, -250E-6, +250E-6);
	TH2D *h2_de_vtx_y_vs_de_xi_56 = new TH2D("h2_de_vtx_y_vs_de_xi_56", ";#Delta#xi^{56};#Deltavtx_{y}^{56}", 50, -5E-3, +5E-3, 50, -250E-6, +250E-6);

	TH2D *h2_de_th_x_vs_de_xi_45 = new TH2D("h2_de_th_x_vs_de_xi_45", ";#Delta#xi^{45};#Deltath_{x}^{45}", 50, -5E-3, +5E-3, 50, -100E-6, +100E-6);
	TH2D *h2_de_th_x_vs_de_xi_56 = new TH2D("h2_de_th_x_vs_de_xi_56", ";#Delta#xi^{56};#Deltath_{x}^{56}", 50, -5E-3, +5E-3, 50, -100E-6, +100E-6);

	TH2D *h2_de_th_y_vs_de_xi_45 = new TH2D("h2_de_th_y_vs_de_xi_45", ";#Delta#xi^{45};#Deltath_{y}^{45}", 50, -5E-3, +5E-3, 50, -100E-6, +100E-6);
	TH2D *h2_de_th_y_vs_de_xi_56 = new TH2D("h2_de_th_y_vs_de_xi_56", ";#Delta#xi^{56};#Deltath_{y}^{56}", 50, -5E-3, +5E-3, 50, -100E-6, +100E-6);

	TH2D *h2_de_vtx_y_vs_de_th_y_45 = new TH2D("h2_de_vtx_y_vs_de_th_y_45", ";#Deltath_{y}^{45};#Deltavtx_{y}^{45}", 50, -100E-6, +100E-6, 50, -250E-6, +250E-6);
	TH2D *h2_de_vtx_y_vs_de_th_y_56 = new TH2D("h2_de_vtx_y_vs_de_th_y_56", ";#Deltath_{y}^{56};#Deltavtx_{y}^{56}", 50, -100E-6, +100E-6, 50, -250E-6, +250E-6);

	// prepare plots - profiles
	TProfile *p_de_vtx_x_vs_xi_45 = new TProfile("p_de_vtx_x_vs_xi_45", ";#xi;#Deltavtx_{x}^{45}", 20, 0., 0.20);
	TProfile *p_de_vtx_x_vs_xi_56 = new TProfile("p_de_vtx_x_vs_xi_56", ";#xi;#Deltavtx_{x}^{56}", 20, 0., 0.20);

	TProfile *p_de_vtx_y_vs_xi_45 = new TProfile("p_de_vtx_y_vs_xi_45", ";#xi;#Deltavtx_{y}^{45}", 20, 0., 0.20);
	TProfile *p_de_vtx_y_vs_xi_56 = new TProfile("p_de_vtx_y_vs_xi_56", ";#xi;#Deltavtx_{y}^{56}", 20, 0., 0.20);

	TProfile *p_de_th_x_vs_xi_45 = new TProfile("p_de_th_x_vs_xi_45", ";#xi;#Deltath_{x}^{45}", 20, 0., 0.20);
	TProfile *p_de_th_x_vs_xi_56 = new TProfile("p_de_th_x_vs_xi_56", ";#xi;#Deltath_{x}^{56}", 20, 0., 0.20);

	TProfile *p_de_th_y_vs_xi_45 = new TProfile("p_de_th_y_vs_xi_45", ";#xi;#Deltath_{y}^{45}", 20, 0., 0.20);
	TProfile *p_de_th_y_vs_xi_56 = new TProfile("p_de_th_y_vs_xi_56", ";#xi;#Deltath_{y}^{56}", 20, 0., 0.20);

	TProfile *p_de_xi_vs_xi_45 = new TProfile("p_de_xi_vs_xi_45", ";#xi;#Delta#xi^{45}", 20, 0., 0.20);
	TProfile *p_de_xi_vs_xi_56 = new TProfile("p_de_xi_vs_xi_56", ";#xi;#Delta#xi^{56}", 20, 0., 0.20);

	// prepare output
	TFile *f_out = TFile::Open(file_output.c_str(), "recreate");

	// event loop
	for (unsigned int evi = 0; evi < n_events; evi++)
	{
		 if (debug)
		 {
			 printf("\n----- event %u -----\n", evi);
		 }

		// generate vertex
		double vtx_x = 0., vtx_y = 0.;

		if (simulate_vertex_x)
			vtx_x += gRandom->Gaus() * beamConditions.si_vtx;

		if (simulate_vertex_y)
			vtx_y += gRandom->Gaus() * beamConditions.si_vtx;

		// generate scattering angles (physics)
		double th_x_45_phys = 0., th_y_45_phys = 0.;
		double th_x_56_phys = 0., th_y_56_phys = 0.;

		if (simulate_scattering_angles_x)
		{
			th_x_45_phys += gRandom->Gaus() * si_th_phys;
			th_x_56_phys += gRandom->Gaus() * si_th_phys;
		}

		if (simulate_scattering_angles_y)
		{
			th_y_45_phys += gRandom->Gaus() * si_th_phys;
			th_y_56_phys += gRandom->Gaus() * si_th_phys;
		}

		// generate beam divergence, calculate complete angle
		double th_x_45 = th_x_45_phys, th_y_45 = th_y_45_phys;
		double th_x_56 = th_x_56_phys, th_y_56 = th_y_56_phys;

		if (simulate_beam_divergence)
		{
			th_x_45 += gRandom->Gaus() * beamConditions.si_beam_div;
			th_y_45 += gRandom->Gaus() * beamConditions.si_beam_div;

			th_x_56 += gRandom->Gaus() * beamConditions.si_beam_div;
			th_y_56 += gRandom->Gaus() * beamConditions.si_beam_div;
		}

		// generate xi
		double xi_45 = 0, xi_56 = 0;
		if (simulate_xi)
		{
			xi_45 = xi_min + gRandom->Rndm() * (xi_max - xi_min);
			xi_56 = xi_min + gRandom->Rndm() * (xi_max - xi_min);
		}

		// print kinematics
		if (debug)
		{
			printf("		vtx_x = %.3f mm, vtx_y = %.3f mm\n", vtx_x*1E3, vtx_y*1E3);
			printf("		th_x_45 = %.1f urad, th_y_45 = %.1f urad, xi_45 = %.3f\n", th_x_45*1E6, th_y_45*1E6, xi_45);
			printf("		th_x_56 = %.1f urad, th_y_56 = %.1f urad, xi_56 = %.3f\n", th_x_56*1E6, th_y_56*1E6, xi_56);
		}

		// proton transport
		TrackDataCollection tracks_45;
		BuildTrackCollection(sector45, vtx_x, vtx_y, th_x_45, th_y_45, xi_45, optics_45, tracks_45);

		TrackDataCollection tracks_56;
		BuildTrackCollection(sector56, vtx_x, vtx_y, th_x_56, th_y_56, xi_56, optics_56, tracks_56);

		// simulate detector resolution
		if (simulate_detector_resolution)
		{
			for (auto &it : tracks_45)
			{
				it.second.x += gRandom->Gaus() * si_det;
				it.second.y += gRandom->Gaus() * si_det;
			}

			for (auto &it : tracks_56)
			{
				it.second.x += gRandom->Gaus() * si_det;
				it.second.y += gRandom->Gaus() * si_det;
			}
		}

		// run reconstruction
		if (debug)
			printf("	reconstruction in 45\n");

		ProtonData proton_45 = protonReconstruction.Reconstruct(sector45, tracks_45);

		if (debug)
		{
			printf("		proton: ");
			proton_45.Print();
		}

		if (debug)
			printf("	reconstruction in 56\n");

		ProtonData proton_56 = protonReconstruction.Reconstruct(sector45, tracks_56);

		if (debug)
		{
			printf("		proton: ");
			proton_56.Print();
		}

		// fill plots
		for (const auto it : tracks_45)
			m_rp_h2_y_vs_x[it.first]->Fill(it.second.x, it.second.y);

		for (const auto it : tracks_56)
			m_rp_h2_y_vs_x[it.first]->Fill(it.second.x, it.second.y);

		if (proton_45.valid)
		{
			const double de_vtx_x = proton_45.vtx_x - vtx_x;
			const double de_vtx_y = proton_45.vtx_y - vtx_y;
			const double de_th_x = proton_45.th_x - th_x_45_phys;
			const double de_th_y = proton_45.th_y - th_y_45_phys;
			const double de_xi = proton_45.xi - xi_45;

			h_de_vtx_x_45->Fill(de_vtx_x);
			h_de_vtx_y_45->Fill(de_vtx_y);
			h_de_th_x_45->Fill(de_th_x);
			h_de_th_y_45->Fill(de_th_y);
			h_de_xi_45->Fill(de_xi);

			h2_de_vtx_x_vs_de_xi_45->Fill(de_xi, de_vtx_x);
			h2_de_vtx_y_vs_de_xi_45->Fill(de_xi, de_vtx_y);
			h2_de_th_x_vs_de_xi_45->Fill(de_xi, de_th_x);
			h2_de_th_y_vs_de_xi_45->Fill(de_xi, de_th_y);
			h2_de_vtx_y_vs_de_th_y_45->Fill(de_th_y, de_vtx_y);

			p_de_vtx_x_vs_xi_45->Fill(xi_45, de_vtx_x);
			p_de_vtx_y_vs_xi_45->Fill(xi_45, de_vtx_y);
			p_de_th_x_vs_xi_45->Fill(xi_45, de_th_x);
			p_de_th_y_vs_xi_45->Fill(xi_45, de_th_y);
			p_de_xi_vs_xi_45->Fill(xi_45, de_xi);
		}

		if (proton_56.valid)
		{
			const double de_vtx_x = proton_56.vtx_x - vtx_x;
			const double de_vtx_y = proton_56.vtx_y - vtx_y;
			const double de_th_x = proton_56.th_x - th_x_56_phys;
			const double de_th_y = proton_56.th_y - th_y_56_phys;
			const double de_xi = proton_56.xi - xi_56;

			h_de_vtx_x_56->Fill(de_vtx_x);
			h_de_vtx_y_56->Fill(de_vtx_y);
			h_de_th_x_56->Fill(de_th_x);
			h_de_th_y_56->Fill(de_th_y);
			h_de_xi_56->Fill(de_xi);

			h2_de_vtx_x_vs_de_xi_56->Fill(de_xi, de_vtx_x);
			h2_de_vtx_y_vs_de_xi_56->Fill(de_xi, de_vtx_y);
			h2_de_th_x_vs_de_xi_56->Fill(de_xi, de_th_x);
			h2_de_th_y_vs_de_xi_56->Fill(de_xi, de_th_y);
			h2_de_vtx_y_vs_de_th_y_56->Fill(de_th_y, de_vtx_y);

			p_de_vtx_x_vs_xi_56->Fill(xi_56, de_vtx_x);
			p_de_vtx_y_vs_xi_56->Fill(xi_56, de_vtx_y);
			p_de_th_x_vs_xi_56->Fill(xi_56, de_th_x);
			p_de_th_y_vs_xi_56->Fill(xi_56, de_th_y);
			p_de_xi_vs_xi_56->Fill(xi_56, de_xi);
		}
	}

	// save plots
	gDirectory = f_out;

	for (const auto &it : m_rp_h2_y_vs_x)
	{
		char buf[100];
		sprintf(buf, "h2_y_vs_x_RP%u", it.first);
		it.second->Write(buf);
	}

	gDirectory = f_out->mkdir("sector 45");
	h_de_vtx_x_45->Write();
	h_de_vtx_y_45->Write();
	h_de_th_x_45->Write();
	h_de_th_y_45->Write();
	h_de_xi_45->Write();

	h2_de_vtx_x_vs_de_xi_45->Write();
	h2_de_vtx_y_vs_de_xi_45->Write();
	h2_de_th_x_vs_de_xi_45->Write();
	h2_de_th_y_vs_de_xi_45->Write();
	h2_de_vtx_y_vs_de_th_y_45->Write();

	p_de_vtx_x_vs_xi_45->Write();
	p_de_vtx_y_vs_xi_45->Write();
	p_de_th_x_vs_xi_45->Write();
	p_de_th_y_vs_xi_45->Write();
	p_de_xi_vs_xi_45->Write();

	ProfileToRMSGraph(p_de_vtx_x_vs_xi_45, "g_rms_de_vtx_x_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_vtx_y_vs_xi_45, "g_rms_de_vtx_y_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_th_x_vs_xi_45, "g_rms_de_th_x_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_th_y_vs_xi_45, "g_rms_de_th_y_vs_xi_45")->Write();
	ProfileToRMSGraph(p_de_xi_vs_xi_45, "g_rms_de_xi_vs_xi_45")->Write();

	gDirectory = f_out->mkdir("sector 56");
	h_de_vtx_x_56->Write();
	h_de_vtx_y_56->Write();
	h_de_th_x_56->Write();
	h_de_th_y_56->Write();
	h_de_xi_56->Write();

	h2_de_vtx_x_vs_de_xi_56->Write();
	h2_de_vtx_y_vs_de_xi_56->Write();
	h2_de_th_x_vs_de_xi_56->Write();
	h2_de_th_y_vs_de_xi_56->Write();
	h2_de_vtx_y_vs_de_th_y_56->Write();

	ProfileToRMSGraph(p_de_vtx_x_vs_xi_56, "g_rms_de_vtx_x_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_vtx_y_vs_xi_56, "g_rms_de_vtx_y_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_th_x_vs_xi_56, "g_rms_de_th_x_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_th_y_vs_xi_56, "g_rms_de_th_y_vs_xi_56")->Write();
	ProfileToRMSGraph(p_de_xi_vs_xi_56, "g_rms_de_xi_vs_xi_56")->Write();

	// clean up
	delete f_out;

	return 0;
}
