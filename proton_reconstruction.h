#ifndef _proton_reconstruction_h_
#define _proton_reconstruction_h_

#include "TFile.h"
#include "TSpline.h"
#include "Fit/Fitter.h"

#include "LHCOpticsApproximator.h"

#include <map>
#include <string>
#include <cmath>

#include "track_lite.h"
#include "beam_conditions.h"

//----------------------------------------------------------------------------------------------------

struct ProtonData
{
	bool valid = false;

	double vtx_x = 0., vtx_y = 0.;	// m

	double th_x = 0., th_y = 0.;	// rad

	double xi = 0.;					// 1, positive when energy loss
	double xi_unc = 0.;

	void Print() const
	{
		printf("valid=%i, vtx_x=%.3f mm, vtx_y=%.3f mm, th_x=%.1f urad, th_y=%.1f urad, xi=%.3f\n",
			valid, vtx_x*1E3, vtx_y*1E3, th_x*1E6, th_y*1E6, xi);
	}
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

class ProtonReconstruction
{
	public:

		ProtonReconstruction() {}

		int Init(const std::string &optics_file_beam1, const std::string &optics_file_beam2);

		ProtonData Reconstruct(LHCSector sector, const TrackDataCollection &tracks) const;

		~ProtonReconstruction()
		{
			for (auto &it : m_rp_optics)
			{
				delete it.second.optics;
				delete it.second.s_xi_vs_x;
				delete it.second.s_y0_vs_xi;
				delete it.second.s_v_y_vs_xi;
				delete it.second.s_L_y_vs_xi;
			}

			if (chiSquareCalculator)
				delete chiSquareCalculator;

			if (fitter)
				delete fitter;
		}

	private:
		/// optics data associated with 1 RP
		struct RPOpticsData
		{
			LHCOpticsApproximator *optics;
			TSpline3 *s_xi_vs_x, *s_y0_vs_xi, *s_v_y_vs_xi, *s_L_y_vs_xi;
		};

		/// map: RP id --> optics data
		std::map<unsigned int, RPOpticsData> m_rp_optics;

		/// class for calculation of chi^2
		class ChiSquareCalculator
		{
			public:
				const TrackDataCollection *tracks;
				const std::map<unsigned int, RPOpticsData> *m_rp_optics;

				ChiSquareCalculator() {}

				double operator() (const double *parameters) const;
		};

		ChiSquareCalculator *chiSquareCalculator = NULL;

		// fitter object
		ROOT::Fit::Fitter *fitter = NULL;
};

//----------------------------------------------------------------------------------------------------

int ProtonReconstruction::Init(const std::string &optics_file_beam1, const std::string &optics_file_beam2)
{
	TFile *f_in_optics_beam1 = TFile::Open(optics_file_beam1.c_str());
	if (f_in_optics_beam1 == NULL)
	{
		printf("ERROR in ProtonReconstruction::Init > Can't open file '%s'.\n", optics_file_beam1.c_str());
		return 1;
	}

	TFile *f_in_optics_beam2 = TFile::Open(optics_file_beam2.c_str());
	if (f_in_optics_beam2 == NULL)
	{
		printf("ERROR in ProtonReconstruction::Init > Can't open file '%s'.\n", optics_file_beam2.c_str());
		return 1;
	}

	// build RP id, optics object name association
	std::map<unsigned int, std::string> nameMap = {
		{2, "ip5_to_station_150_h_1_lhcb2"},
		{3, "ip5_to_station_150_h_2_lhcb2"},
		{102, "ip5_to_station_150_h_1_lhcb1"},
		{103, "ip5_to_station_150_h_2_lhcb1"},
	};

	// build optics data for each object
	for (const auto &it : nameMap)
	{
		const unsigned int &rpId = it.first;
		const std::string &ofName = it.second;

		// determine LHC sector from RP id
		LHCSector sector = unknownSector;
		if ((rpId / 100) == 0)
			sector = sector45;
		if ((rpId / 100) == 1)
			sector = sector56;

		// load optics approximation
		TFile *f_in_optics = NULL;
		if (sector == sector45)
			f_in_optics = f_in_optics_beam2;
		if (sector == sector56)
			f_in_optics = f_in_optics_beam1;

		LHCOpticsApproximator *of_orig = (LHCOpticsApproximator *) f_in_optics->Get(ofName.c_str());

		if (of_orig == NULL)
		{
			printf("ERROR in ProtonReconstruction::Init > Can't load object '%s'.\n", ofName.c_str());
			return 2;
		}

		RPOpticsData rpod;
		rpod.optics = new LHCOpticsApproximator(* of_orig);

		// build auxiliary optical functions
		double crossing_angle = 0.;
		if (sector == sector45)
			crossing_angle = beamConditions.half_crossing_angle_45;
		if (sector == sector56)
			crossing_angle = beamConditions.half_crossing_angle_56;

		const bool check_appertures = false;
		const bool invert_beam_coord_sytems = true;

		TGraph *g_xi_vs_x = new TGraph();
		TGraph *g_y0_vs_xi = new TGraph();
		TGraph *g_v_y_vs_xi = new TGraph();
		TGraph *g_L_y_vs_xi = new TGraph();

		for (double xi = 0.; xi <= 0.201; xi += 0.005)
		{
			// input: only xi
			double kin_in_xi[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y, 0., -xi };
			double kin_out_xi[5];
    		rpod.optics->Transport(kin_in_xi, kin_out_xi, check_appertures, invert_beam_coord_sytems);

			// input: xi and vtx_y
			const double vtx_y = 10E-6;	// m
			double kin_in_xi_vtx_y[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y + vtx_y, 0., -xi };
			double kin_out_xi_vtx_y[5];
    		rpod.optics->Transport(kin_in_xi_vtx_y, kin_out_xi_vtx_y, check_appertures, invert_beam_coord_sytems);

			// input: xi and th_y
			const double th_y = 20E-6;	// rad
			double kin_in_xi_th_y[5] = { 0., crossing_angle * (1. - xi), beamConditions.vtx0_y, th_y * (1. - xi), -xi };
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
	chiSquareCalculator = new ChiSquareCalculator();

	fitter = new ROOT::Fit::Fitter();
	double pStart[] = {0, 0, 0, 0};
	fitter->SetFCN(4, *chiSquareCalculator, pStart, 0, true);

	// clean up
	delete f_in_optics_beam1;
	delete f_in_optics_beam2;

	return 0;
}

//----------------------------------------------------------------------------------------------------

double ProtonReconstruction::ChiSquareCalculator::operator() (const double *parameters) const
{
	// extract proton parameters
	const double &xi = parameters[0];
	const double &th_x = parameters[1];
	const double &th_y = parameters[2];
	const double vtx_x = 0;
	const double &vtx_y = parameters[3];

	// calculate chi^2
	double S2 = 0.;

	for (auto &it : *tracks)
	{
		const unsigned int &rpId = it.first;

		// determine LHC sector from RP id
		LHCSector sector = unknownSector;
		if ((rpId / 100) == 0)
			sector = sector45;
		if ((rpId / 100) == 1)
			sector = sector56;

		double crossing_angle = 0.;
		if (sector == sector45)
			crossing_angle = beamConditions.half_crossing_angle_45;
		if (sector == sector56)
			crossing_angle = beamConditions.half_crossing_angle_56;

		// transport proton to the RP
		auto oit = m_rp_optics->find(rpId);

		const bool check_appertures = false;
		const bool invert_beam_coord_sytems = true;

		double kin_in[5] = {
			vtx_x,
			(th_x + crossing_angle) * (1. - xi),
			beamConditions.vtx0_y + vtx_y,
			th_y * (1. - xi),
			-xi
		};
		double kin_out[5];
    	oit->second.optics->Transport(kin_in, kin_out, check_appertures, invert_beam_coord_sytems);

		const double &x = kin_out[0];
		const double &y = kin_out[2];

		// calculate chi^2 constributions
		const double x_diff_norm = (x - it.second.x) / it.second.x_unc;
		const double y_diff_norm = (y - it.second.y) / it.second.y_unc;

		// increase chi^2
		S2 += x_diff_norm*x_diff_norm + y_diff_norm*y_diff_norm;
	}

	//printf("xi=%.3E, th_x=%.3E, th_y=%.3E, vtx_y=%.3E | S2 = %.3E\n", xi, th_x, th_y, vtx_y, S2);

	return S2;
}

//----------------------------------------------------------------------------------------------------

ProtonData ProtonReconstruction::Reconstruct(LHCSector /* sector */, const TrackDataCollection &tracks) const
{
	// by default invalid proton
	ProtonData pd;
	pd.valid = false;

	// need at least two tracks
	if (tracks.size() < 2)
		return pd;

	// check optics is available for all tracks
	for (const auto &it : tracks)
	{
		auto oit = m_rp_optics.find(it.first);
		if (oit == m_rp_optics.end())
		{
			printf("ERROR in ProtonReconstruction::Reconstruct > optics not available for RP %u.\n", it.first);
			return pd;
		}
	}

	// rough estimate of xi
	double S_xi0 = 0., S_1 = 0.;
	for (const auto &it : tracks)
	{
		auto oit = m_rp_optics.find(it.first);
		double xi = oit->second.s_xi_vs_x->Eval(it.second.x);

		S_1 += 1.;
		S_xi0 += xi;
	}

	const double xi_0 = S_xi0 / S_1;

	//printf("    xi_0 = %.3f\n", xi_0);

	// rough estimate of th_y and vtx_y
	double y[2], v_y[2], L_y[2];
	unsigned int y_idx = 0;
	for (const auto &it : tracks)
	{
		if (y_idx >= 2)
			continue;

		auto oit = m_rp_optics.find(it.first);

		y[y_idx] = it.second.y - oit->second.s_y0_vs_xi->Eval(xi_0);
		v_y[y_idx] = oit->second.s_v_y_vs_xi->Eval(xi_0);
		L_y[y_idx] = oit->second.s_L_y_vs_xi->Eval(xi_0);

		y_idx++;
	}

	const double det = v_y[0] * L_y[1] - L_y[0] * v_y[1];
	const double vtx_y_0 = (L_y[1] * y[0] - L_y[0] * y[1]) / det;
	const double th_y_0 = (v_y[0] * y[1] - v_y[1] * y[0]) / det;

	//printf("    vtx_y_0 = %.3f mm\n", vtx_y_0 * 1E3);
	//printf("    th_y_0 = %.1f urad\n", th_y_0 * 1E6);

	// minimisation
	fitter->Config().ParSettings(0).Set("xi", xi_0, 0.005);
	fitter->Config().ParSettings(1).Set("th_x", 0., 20E-6);
	fitter->Config().ParSettings(2).Set("th_y", th_y_0, 1E-6);
	fitter->Config().ParSettings(3).Set("vtx_y", vtx_y_0, 1E-6);

	// TODO: this breaks the const-ness ??
	chiSquareCalculator->tracks = &tracks;
	chiSquareCalculator->m_rp_optics = &m_rp_optics;

	fitter->FitFCN();

	// extract proton parameters
	const ROOT::Fit::FitResult &result = fitter->Result();
	const double *fitParameters = result.GetParams();

	pd.valid = true;
	pd.vtx_x = 0.;
	pd.vtx_y = fitParameters[3];
	pd.th_x = fitParameters[1];
	pd.th_y = fitParameters[2];
	pd.xi = fitParameters[0];

	return pd;
}

#endif
