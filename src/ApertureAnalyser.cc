#include "ApertureAnalyser.h"
#include "LHCOpticsApproximator.h"
#include "TFile.h"
#include "TMath.h"
#include "vector"
#include "set"
#include "TRandom2.h"
#include <ostream>
#include <fstream>
#include <map>

void ApertureHistogramsEntity::PreprocessBeforeWriting() {
	t_xi_absorbtion_acceptance_in_selected_sequence.Divide(
			&t_xi_absorbed_in_selected_sequence, &t_xi_oryginal_dist);
	t_xi_absorbtion_acceptance_in_single_aperture.Divide(
			&t_xi_absorbed_in_single_aperture, &t_xi_oryginal_dist);

	t_xi_surviving_tracks_after_single_aperture_acceptance.Divide(
			&t_xi_surviving_tracks_after_single_aperture, &t_xi_oryginal_dist);
	t_xi_surviving_tracks_after_selected_sequence_acceptance.Divide(
			&t_xi_surviving_tracks_after_selected_sequence,
			&t_xi_oryginal_dist);
	t_xi_tracks_lost_in_selected_sequence_acceptance.Divide(
			&t_xi_tracks_lost_in_selected_sequence, &t_xi_oryginal_dist);
}

bool ApertureHistogramsEntity::CheckIfPhysicalProton(double beam_energy,
		double ksi, double t) {
	long double m0_ = proton_mass_;
	long double p1_ = TMath::Sqrt(beam_energy * beam_energy - m0_ * m0_);

	long double term1 = 1 + ksi;
	long double term2 = term1 * term1;
	long double term3 = m0_ * m0_;
	long double term4 = p1_ * p1_;
	long double term5 = term3 * term3;
	long double term6 = ksi * ksi;
	long double sqrt1 = TMath::Sqrt(term3 + term4);
	long double sqrt2 = TMath::Sqrt(term3 + term2 * term4);
	long double denom1 = term2 * term4 * term4;
	long double denom2 = term2 * term4;

	long double bracket1 = -2 * term5 / denom1 - 2 * term3 / denom2
			- 2 * ksi * term3 / denom2 - term6 * term3 / denom2
			+ 2 * term3 * sqrt1 * sqrt2 / denom1;

	long double bracket2 = t
			* (term3 / denom1 - t / (4 * denom1) - sqrt1 * sqrt2 / denom1);

	long double theta_squared = bracket1 + bracket2;

	bool res = false;
	long double theta = 0.0;

	if (theta_squared >= 0.0 && theta_squared < 0.01) {
		theta = TMath::Sqrt(theta_squared);
		res = true;
	}

	res = res || (ksi == 0.0 && t == 0.0);
	return res;
}

double ApertureHistogramsEntity::IPSmearedProtonMomentumTot(double px,
		double py, double pz) const //GeV
		{
	long double part_en = TMath::Sqrt(
			px * px + py * py + pz * pz + proton_mass_ * proton_mass_);
	long double energy_diff = part_en - beam_energy_;
	long double px_diff = px - nominal_beam1_px_smeared_;
	long double py_diff = py - nominal_beam1_py_smeared_;
	long double pz_diff = TMath::Abs(pz) - nominal_beam1_pz_smeared_;

	long double t = energy_diff * energy_diff - px_diff * px_diff
			- py_diff * py_diff - pz_diff * pz_diff;
	return t;
}

double ApertureHistogramsEntity::CanonicalAnglesTot(double ThetaX,
		double ThetaY, double Xi) const //GeV
		{
	long double px = ThetaX * beam_momentum_;
	long double py = ThetaY * beam_momentum_;
	long double part_momentum = beam_momentum_ * (1.0 + Xi);
	long double pz = TMath::Sqrt(
			part_momentum * part_momentum - px * px - py * py);

	return IPSmearedProtonMomentumTot(px, py, pz);
}

void ApertureHistogramsEntity::ComputeOpticsParameters(
		ApertureAnalysisConf &conf) {
	proton_mass_ = 0.93827201323;
	beam_energy_ = conf.nominal_beam_energy;
	beam_momentum_ = TMath::Sqrt(
			beam_energy_ * beam_energy_ - proton_mass_ * proton_mass_); //[GeV]
	nominal_beam1_px_smeared_ = beam_momentum_
			* TMath::Sin(conf.x_half_crossing_angle);
	nominal_beam1_py_smeared_ = beam_momentum_
			* TMath::Sin(conf.y_half_crossing_angle);
	double beam1_theta = TMath::Sqrt(
			conf.x_half_crossing_angle * conf.x_half_crossing_angle
					+ conf.y_half_crossing_angle * conf.y_half_crossing_angle);
	nominal_beam1_pz_smeared_ = beam_momentum_ * TMath::Cos(beam1_theta);
}

void ApertureHistogramsEntity::FillApertureHitPositionSelectedSequence(
		MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot,
		ApertureAnalysisConf &conf) {
	aperture_hits_in_selected_squence.Fill(out_prot.x, out_prot.y);
}

void ApertureHistogramsEntity::FillApertureHitInfoSelectedSequence(
		MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf) {
	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (t < 0.0 && xi <= 0.0
			&& CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_absorbed_in_selected_sequence.Fill(log10_t, log10_xi);
	}
}

void ApertureHistogramsEntity::FillApertureHitPositionSingleAperture(
		MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot,
		ApertureAnalysisConf &conf) {
	aperture_hits_in_single_aperture.Fill(out_prot.x, out_prot.y);
}

void ApertureHistogramsEntity::FillApertureHitInfoSingleAperture(
		MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf) {
	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (t < 0.0 && xi <= 0.0
			&& CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_absorbed_in_single_aperture.Fill(log10_t, log10_xi);

		if (log10_t < -1.65 && log10_xi < -1.25) {
			std::cout << "in_prot.theta_x=" << in_prot.theta_x
					<< " in_prot.theta_y=" << in_prot.theta_y << " in_prot.ksi="
					<< in_prot.ksi << "in_prot.x=" << in_prot.x << " in_prot.y="
					<< in_prot.y << std::endl;
		}
	}
}

void ApertureHistogramsEntity::FillSingleApertureSurvived(
		MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot,
		bool out_pos_valid, ApertureAnalysisConf &conf) {
	if (out_pos_valid) {
		debug_x_y_surviving_tracks_at_target_after_single_aperture.Fill(
				out_prot.x, out_prot.y);
	}

	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (t < 0.0 && xi <= 0.0
			&& CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_surviving_tracks_after_single_aperture.Fill(log10_t, log10_xi);
	}
}

void ApertureHistogramsEntity::FillTrackSurvivedSelectedApertures(
		MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot,
		bool out_pos_valid, ApertureAnalysisConf &conf) {
	if (out_pos_valid) {
		debug_x_y_surviving_tracks_at_target_after_selected_sequence.Fill(
				out_prot.x, out_prot.y);
	}

	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (t < 0.0 && xi <= 0.0
			&& CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_surviving_tracks_after_selected_sequence.Fill(log10_t, log10_xi);
	}
}

void ApertureHistogramsEntity::FillTrackLostInSelectedApertures(
		MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot,
		bool out_pos_valid, ApertureAnalysisConf &conf) {
	if (out_pos_valid) {
		debug_x_y_tracks_at_target_lost_in_selected_sequence.Fill(out_prot.x,
				out_prot.y);
	}

	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (t < 0.0 && xi <= 0.0
			&& CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_tracks_lost_in_selected_sequence.Fill(log10_t, log10_xi);
	}
}

void ApertureHistogramsEntity::FillReferenceHists(
		MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf) {
	double t = CanonicalAnglesTot(in_prot.theta_x, in_prot.theta_y,
			in_prot.ksi);
	double xi = in_prot.ksi;

	if (CheckIfPhysicalProton(conf.nominal_beam_energy, xi, t)) {
		double log10_t = TMath::Log10(TMath::Abs(t));
		double log10_xi = TMath::Log10(TMath::Abs(xi));
		t_xi_oryginal_dist.Fill(log10_t, log10_xi);
	}

	debug_thx_thy_oryginal_dist.Fill(in_prot.theta_x, in_prot.theta_y);
	debug_x_y_oryginal_dist.Fill(in_prot.x, in_prot.y);
	debug_xi_oryginal_dist.Fill(in_prot.ksi);
}

ApertureHistogramsEntity::ApertureHistogramsEntity(std::string aperture_name,
		ApertureAnalysisConf &conf) {
	aperture_name_ = aperture_name;
	ComputeOpticsParameters(conf);
	AllocateHistograms(conf);
}

void ApertureHistogramsEntity::AddNamePrefix(std::string prefix) {
	aperture_name_ = prefix + aperture_name_;
}

void ApertureHistogramsEntity::AllocateHistograms(ApertureAnalysisConf &conf) {
	double log_t_min = TMath::Log10(TMath::Abs(conf.t_min));
	double log_t_max = TMath::Log10(TMath::Abs(conf.t_max));
	double log_xi_min = TMath::Log10(TMath::Abs(conf.xi_min));
	double log_xi_max = TMath::Log10(TMath::Abs(/*conf.xi_max*/1.0));

	int t_bins = 40;
	int xi_bins = 40;
	int aper_bins = 100;
	double aper_x_min = -0.05;
	double aper_x_max = 0.05;
	double aper_y_min = -0.05;
	double aper_y_max = 0.05;

	std::string name;

	name = aperture_name_
			+ std::string("_01_aperture_loss_increment_in_selected_squence");
	aperture_hits_in_selected_squence = TH2F(name.c_str(), name.c_str(),
			aper_bins, aper_x_min, aper_x_max, aper_bins, aper_y_min,
			aper_y_max);
	aperture_hits_in_selected_squence.SetDirectory(0);
	aperture_hits_in_selected_squence.SetXTitle("x [m]");
	aperture_hits_in_selected_squence.SetYTitle("y [m]");

	name = aperture_name_ + std::string("_16_t_xi_oryginal_dist");
	t_xi_oryginal_dist = TH2F(name.c_str(), name.c_str(), t_bins, log_t_min,
			log_t_max, xi_bins, log_xi_min, log_xi_max);
	t_xi_oryginal_dist.SetDirectory(0);
	t_xi_oryginal_dist.SetXTitle("Log(-t/GeV^{2})");
	t_xi_oryginal_dist.SetYTitle("Log(-#xi)");

	name = aperture_name_
			+ std::string("_02_t_xi_loss_increment_in_selected_sequence");
	t_xi_absorbed_in_selected_sequence = TH2F(name.c_str(), name.c_str(),
			t_bins, log_t_min, log_t_max, xi_bins, log_xi_min, log_xi_max);
	t_xi_absorbed_in_selected_sequence.SetDirectory(0);
	t_xi_absorbed_in_selected_sequence.SetXTitle("Log(-t/GeV^{2})");
	t_xi_absorbed_in_selected_sequence.SetYTitle("Log(-#xi)");

	name = aperture_name_
			+ std::string(
					"_03_t_xi_loss_increment_acceptance_in_selected_sequence");
	t_xi_absorbtion_acceptance_in_selected_sequence = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_absorbtion_acceptance_in_selected_sequence.SetDirectory(0);
	t_xi_absorbtion_acceptance_in_selected_sequence.SetXTitle(
			"Log(-t/GeV^{2})");
	t_xi_absorbtion_acceptance_in_selected_sequence.SetYTitle("Log(-#xi)");

	name = aperture_name_ + std::string("_04_proton_losses_in_aperture_alone");
	aperture_hits_in_single_aperture = TH2F(name.c_str(), name.c_str(),
			aper_bins, aper_x_min, aper_x_max, aper_bins, aper_y_min,
			aper_y_max);
	aperture_hits_in_single_aperture.SetDirectory(0);
	aperture_hits_in_single_aperture.SetXTitle("x [m]");
	aperture_hits_in_single_aperture.SetYTitle("y [m]");

	name = aperture_name_ + std::string("_05_t_xi_losses_in_aperture_alone");
	t_xi_absorbed_in_single_aperture = TH2F(name.c_str(), name.c_str(), t_bins,
			log_t_min, log_t_max, xi_bins, log_xi_min, log_xi_max);
	t_xi_absorbed_in_single_aperture.SetDirectory(0);
	t_xi_absorbed_in_single_aperture.SetXTitle("Log(-t/GeV^{2})");
	t_xi_absorbed_in_single_aperture.SetYTitle("Log(-#xi)");

	name = aperture_name_
			+ std::string("_06_t_xi_losses_acceptance_in_aperture_alone");
	t_xi_absorbtion_acceptance_in_single_aperture = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_absorbtion_acceptance_in_single_aperture.SetDirectory(0);
	t_xi_absorbtion_acceptance_in_single_aperture.SetXTitle("Log(-t/GeV^{2})");
	t_xi_absorbtion_acceptance_in_single_aperture.SetYTitle("Log(-#xi)");

	name = aperture_name_ + std::string("_18_debug_thx_thy_oryginal_dist");
	debug_thx_thy_oryginal_dist = TH2F(name.c_str(), name.c_str(), 100, 0, 1e-6,
			100, 0, 1e-6);
	debug_thx_thy_oryginal_dist.SetDirectory(0);
	debug_thx_thy_oryginal_dist.SetBit(TH2::kCanRebin);
	debug_thx_thy_oryginal_dist.SetXTitle("#Theta_{x} [rad]");
	debug_thx_thy_oryginal_dist.SetYTitle("#Theta_{y} [rad]");

	name = aperture_name_ + std::string("_17_debug_x_y_oryginal_dist");
	debug_x_y_oryginal_dist = TH2F(name.c_str(), name.c_str(), 100, 0, 1e-6,
			100, 0, 1e-6);
	debug_x_y_oryginal_dist.SetDirectory(0);
	debug_x_y_oryginal_dist.SetBit(TH2::kCanRebin);
	debug_x_y_oryginal_dist.SetXTitle("x [m]");
	debug_x_y_oryginal_dist.SetYTitle("y [m]");

	name = aperture_name_ + std::string("_19_debug_xi_oryginal_dist");
	debug_xi_oryginal_dist = TH1F(name.c_str(), name.c_str(), 100, 0, 1e-6);
	debug_xi_oryginal_dist.SetDirectory(0);
	debug_xi_oryginal_dist.SetBit(TH1::kCanRebin);
	debug_xi_oryginal_dist.SetXTitle("#xi");
	debug_xi_oryginal_dist.SetYTitle("Entries");

	name = aperture_name_
			+ std::string("_07_t_xi_surviving_tracks_after_single_aperture");
	t_xi_surviving_tracks_after_single_aperture = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_surviving_tracks_after_single_aperture.SetDirectory(0);
	t_xi_surviving_tracks_after_single_aperture.SetXTitle("Log(-t/GeV^{2})");
	t_xi_surviving_tracks_after_single_aperture.SetYTitle("Log(-#xi)");

	name =
			aperture_name_
					+ std::string(
							"_08_t_xi_surviving_tracks_after_single_aperture_acceptance");
	t_xi_surviving_tracks_after_single_aperture_acceptance = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_surviving_tracks_after_single_aperture_acceptance.SetDirectory(0);
	t_xi_surviving_tracks_after_single_aperture_acceptance.SetXTitle(
			"Log(-t/GeV^{2})");
	t_xi_surviving_tracks_after_single_aperture_acceptance.SetYTitle(
			"Log(-#xi)");

	name =
			aperture_name_
					+ std::string(
							"_09_debug_x_y_surviving_tracks_at_target_after_single_aperture");
	debug_x_y_surviving_tracks_at_target_after_single_aperture = TH2F(
			name.c_str(), name.c_str(), 100, -0.05, 0.05, 100, -0.05, 0.05);
	debug_x_y_surviving_tracks_at_target_after_single_aperture.SetDirectory(0);
	debug_x_y_surviving_tracks_at_target_after_single_aperture.SetXTitle(
			"x [m]");
	debug_x_y_surviving_tracks_at_target_after_single_aperture.SetYTitle(
			"y [m]");

	name = aperture_name_
			+ std::string("_10_t_xi_surviving_tracks_after_selected_sequence");
	t_xi_surviving_tracks_after_selected_sequence = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_surviving_tracks_after_selected_sequence.SetDirectory(0);
	t_xi_surviving_tracks_after_selected_sequence.SetXTitle("Log(-t/GeV^{2})");
	t_xi_surviving_tracks_after_selected_sequence.SetYTitle("Log(-#xi)");

	name =
			aperture_name_
					+ std::string(
							"_11_t_xi_surviving_tracks_after_selected_sequence_acceptance");
	t_xi_surviving_tracks_after_selected_sequence_acceptance = TH2F(
			name.c_str(), name.c_str(), t_bins, log_t_min, log_t_max, xi_bins,
			log_xi_min, log_xi_max);
	t_xi_surviving_tracks_after_selected_sequence_acceptance.SetDirectory(0);
	t_xi_surviving_tracks_after_selected_sequence_acceptance.SetXTitle(
			"Log(-t/GeV^{2})");
	t_xi_surviving_tracks_after_selected_sequence_acceptance.SetYTitle(
			"Log(-#xi)");

	name =
			aperture_name_
					+ std::string(
							"_12_debug_x_y_surviving_tracks_at_target_after_selected_sequence");
	debug_x_y_surviving_tracks_at_target_after_selected_sequence = TH2F(
			name.c_str(), name.c_str(), 100, -0.05, 0.05, 100, -0.05, 0.05);
	debug_x_y_surviving_tracks_at_target_after_selected_sequence.SetDirectory(
			0);
//  .SetBit(TH2::kCanRebin);
	debug_x_y_surviving_tracks_at_target_after_selected_sequence.SetXTitle(
			"x [m]");
	debug_x_y_surviving_tracks_at_target_after_selected_sequence.SetYTitle(
			"y [m]");

	name = aperture_name_
			+ std::string("_13_t_xi_tracks_lost_in_selected_sequence");
	t_xi_tracks_lost_in_selected_sequence = TH2F(name.c_str(), name.c_str(),
			t_bins, log_t_min, log_t_max, xi_bins, log_xi_min, log_xi_max);
	t_xi_tracks_lost_in_selected_sequence.SetDirectory(0);
	t_xi_tracks_lost_in_selected_sequence.SetXTitle("Log(-t/GeV^{2})");
	t_xi_tracks_lost_in_selected_sequence.SetYTitle("Log(-#xi)");

	name = aperture_name_
			+ std::string(
					"_14_t_xi_tracks_lost_in_selected_sequence_acceptance");
	t_xi_tracks_lost_in_selected_sequence_acceptance = TH2F(name.c_str(),
			name.c_str(), t_bins, log_t_min, log_t_max, xi_bins, log_xi_min,
			log_xi_max);
	t_xi_tracks_lost_in_selected_sequence_acceptance.SetDirectory(0);
	t_xi_tracks_lost_in_selected_sequence_acceptance.SetXTitle(
			"Log(-t/GeV^{2})");
	t_xi_tracks_lost_in_selected_sequence_acceptance.SetYTitle("Log(-#xi)");

	name = aperture_name_
			+ std::string(
					"_15_debug_x_y_tracks_at_target_lost_in_selected_sequence");
	debug_x_y_tracks_at_target_lost_in_selected_sequence = TH2F(name.c_str(),
			name.c_str(), 100, -0.05, 0.05, 100, -0.05, 0.05);
	debug_x_y_tracks_at_target_lost_in_selected_sequence.SetDirectory(0);
	debug_x_y_tracks_at_target_lost_in_selected_sequence.SetXTitle("x [m]");
	debug_x_y_tracks_at_target_lost_in_selected_sequence.SetYTitle("y [m]");
}

void ApertureHistogramsEntity::Write() {
	PreprocessBeforeWriting();

	gDirectory->mkdir(aperture_name_.c_str());
	gDirectory->cd(aperture_name_.c_str());

	aperture_hits_in_selected_squence.Write("", TObject::kWriteDelete);
	t_xi_oryginal_dist.Write("", TObject::kWriteDelete);
	t_xi_absorbed_in_selected_sequence.Write("", TObject::kWriteDelete);
	t_xi_absorbtion_acceptance_in_selected_sequence.Write("",
			TObject::kWriteDelete);

	aperture_hits_in_single_aperture.Write("", TObject::kWriteDelete);
	t_xi_absorbed_in_single_aperture.Write("", TObject::kWriteDelete);
	t_xi_absorbtion_acceptance_in_single_aperture.Write("",
			TObject::kWriteDelete);

	debug_thx_thy_oryginal_dist.Write("", TObject::kWriteDelete);
	debug_x_y_oryginal_dist.Write("", TObject::kWriteDelete);
	debug_xi_oryginal_dist.Write("", TObject::kWriteDelete);

	t_xi_surviving_tracks_after_single_aperture.Write("",
			TObject::kWriteDelete);
	t_xi_surviving_tracks_after_single_aperture_acceptance.Write("",
			TObject::kWriteDelete);
	debug_x_y_surviving_tracks_at_target_after_single_aperture.Write("",
			TObject::kWriteDelete);

	t_xi_surviving_tracks_after_selected_sequence.Write("",
			TObject::kWriteDelete);
	t_xi_surviving_tracks_after_selected_sequence_acceptance.Write("",
			TObject::kWriteDelete);
	debug_x_y_surviving_tracks_at_target_after_selected_sequence.Write("",
			TObject::kWriteDelete);

	t_xi_tracks_lost_in_selected_sequence.Write("", TObject::kWriteDelete);
	t_xi_tracks_lost_in_selected_sequence_acceptance.Write("",
			TObject::kWriteDelete);
	debug_x_y_tracks_at_target_lost_in_selected_sequence.Write("",
			TObject::kWriteDelete);

	gDirectory->cd("..");
}

ApertureAnalyser::ApertureAnalyser(std::string conf_file_name) {
	conf_file_name_ = conf_file_name;
	OpenXMLConfigurationFile(conf_file_name_);

	std::vector<int> ids = xml_parser_.getIds();
	for (int i = 0; i < ids.size(); i++) {
		aperture_tracks_.clear();
		aperture_hists_.clear();
		all_aperture_hists_.clear();
		conf_ = GetParamConfiguration(ids[i]);
		ReadParameterisation(conf_);
		AnalyseApertures();
	}
}

void ApertureAnalyser::OpenXMLConfigurationFile(std::string file_name) {
	xml_parser_.read(file_name);
	xml_parser_.setFilename(file_name);
	xml_parser_.read();
}

ApertureAnalysisConf ApertureAnalyser::GetParamConfiguration(int id) {
	ApertureAnalysisConf conf;

	conf.optics_apperture_parametrisation_file = xml_parser_.get < std::string
			> (id, std::string("optics_apperture_parametrisation_file"));
	conf.optics_apperture_parametrisation_name = xml_parser_.get < std::string
			> (id, std::string("optics_apperture_parametrisation_name"));
	conf.analysis_output_file = xml_parser_.get < std::string
			> (id, std::string("analysis_output_file"));
	conf.analysis_output_hist_file = xml_parser_.get < std::string
			> (id, std::string("analysis_output_hist_file"));

	conf.t_min = xml_parser_.get<double>(id, std::string("t_min"));
	conf.t_max = xml_parser_.get<double>(id, std::string("t_max"));
	conf.xi_min = xml_parser_.get<double>(id, std::string("xi_min"));
	conf.xi_max = xml_parser_.get<double>(id, std::string("xi_max"));
	conf.ip_beta = xml_parser_.get<double>(id, std::string("ip_beta"));
	conf.ip_norm_emit = xml_parser_.get<double>(id,
			std::string("ip_norm_emit"));
	conf.nominal_beam_energy = xml_parser_.get<double>(id,
			std::string("nominal_beam_energy"));

	conf.x_offset = xml_parser_.get<double>(id, std::string("x_offset"));
	conf.x_half_crossing_angle = xml_parser_.get<double>(id,
			std::string("x_half_crossing_angle"));
	conf.y_offset = xml_parser_.get<double>(id, std::string("y_offset"));
	conf.y_half_crossing_angle = xml_parser_.get<double>(id,
			std::string("y_half_crossing_angle"));
	conf.random_seed = xml_parser_.get<int>(id, std::string("random_seed"));

	conf.lost_smaple_population = xml_parser_.get<int>(id,
			std::string("lost_smaple_population"));

	return conf;
}

bool ApertureAnalyser::ReadParameterisation(ApertureAnalysisConf &conf) {
	TFile *f = TFile::Open(conf.optics_apperture_parametrisation_file.c_str());
	LHCOpticsApproximator * approx_ptr = (LHCOpticsApproximator*) f->Get(
			conf.optics_apperture_parametrisation_name.c_str());
	parameterisation_ = (*approx_ptr);
	apertures_ = parameterisation_.GetApertures();
	f->Close();
	return true;
}

void ApertureAnalyser::Sort(double &max, double &min) {
	if (max >= min)
		return;
	else {
		double temp = max;
		max = min;
		min = temp;
	}
}

void ApertureAnalyser::AnalyseApertures() {
	OpenInfoTextFile(conf_);
	std::cout << "There is " << apertures_.size() << " apertures to analyse"
			<< std::endl;
	out_file_ << "There is " << apertures_.size() << " apertures to analyse"
			<< std::endl;
	BuildTestTracks(conf_);
	AnalyseApertureLosses(conf_);
	AnalyseAllApertureAcceptances(conf_);
	FindTheBottleneckApertures(conf_);
	AnalyseDestinationAcceptance(conf_);
	CloseInfoTextFile(conf_);
	WriteApertureHistograms(conf_);
}

void ApertureAnalyser::BuildTestTracks(ApertureAnalysisConf &conf_) {
	int number_of_samples = conf_.lost_smaple_population;
	double theta_max = TMath::Sqrt(
			TMath::Abs(conf_.t_max)
					/ (conf_.nominal_beam_energy * conf_.nominal_beam_energy));
	double theta_min = TMath::Sqrt(
			TMath::Abs(conf_.t_min)
					/ (conf_.nominal_beam_energy * conf_.nominal_beam_energy));
	Sort(theta_max, theta_min);

	double xi_min = TMath::Abs(conf_.xi_min);
	double xi_max = TMath::Abs(conf_.xi_max);
	Sort(xi_max, xi_min);

	double lorentz_gammma = conf_.nominal_beam_energy / 0.93827201323;
	double beam_size = TMath::Sqrt(
			conf_.ip_beta * conf_.ip_norm_emit / lorentz_gammma);

	TRandom2 rand(conf_.random_seed);

	double log_th_min = TMath::Log(theta_min);
	double log_th_max = TMath::Log(theta_max);
	double log_xi_min = TMath::Log(xi_min);
	double log_xi_max = TMath::Log(xi_max);

	std::cout << "Generating tracks for the following configuration:"
			<< std::endl;
	std::cout << "=================================================="
			<< std::endl;
	std::cout << "beam_size=" << beam_size << std::endl;
	std::cout << "beam x offset=" << conf_.x_offset << std::endl;
	std::cout << "beam y offset=" << conf_.y_offset << std::endl;
	std::cout << "x_half_crossing_angle=" << conf_.x_half_crossing_angle
			<< std::endl;
	std::cout << "y_half_crossing_angle=" << conf_.y_half_crossing_angle
			<< std::endl;
	std::cout << "theta_min=" << theta_min << std::endl;
	std::cout << "theta_max=" << theta_max << std::endl;
	std::cout << "xi_min=" << -xi_min << std::endl;
	std::cout << "xi_max=" << -xi_max << std::endl << std::endl << std::endl;
	std::cout << "optics_apperture_parametrisation_file="
			<< conf_.optics_apperture_parametrisation_file << std::endl;
	std::cout << "optics_apperture_parametrisation_name="
			<< conf_.optics_apperture_parametrisation_name << std::endl;
	std::cout << "lost_smaple_population=" << conf_.lost_smaple_population
			<< std::endl;

	out_file_ << "Generating tracks for the following configuration:"
			<< std::endl;
	out_file_ << "=================================================="
			<< std::endl;
	out_file_ << "beam_size=" << beam_size << std::endl;
	out_file_ << "beam x offset=" << conf_.x_offset << std::endl;
	out_file_ << "beam y offset=" << conf_.y_offset << std::endl;
	out_file_ << "x_half_crossing_angle=" << conf_.x_half_crossing_angle
			<< std::endl;
	out_file_ << "y_half_crossing_angle=" << conf_.y_half_crossing_angle
			<< std::endl;
	out_file_ << "theta_min=" << theta_min << std::endl;
	out_file_ << "theta_max=" << theta_max << std::endl;
	out_file_ << "xi_min=" << -xi_min << std::endl;
	out_file_ << "xi_max=" << -xi_max << std::endl << std::endl << std::endl;
	out_file_ << "optics_apperture_parametrisation_file="
			<< conf_.optics_apperture_parametrisation_file << std::endl;
	out_file_ << "optics_apperture_parametrisation_name="
			<< conf_.optics_apperture_parametrisation_name << std::endl;
	out_file_ << "lost_smaple_population=" << conf_.lost_smaple_population
			<< std::endl;

	int track_counter = 0;
	int total_track_counter = 0;
	while (track_counter < conf_.lost_smaple_population) {
		double x;
		double y;
		do {
			//[m], [rad]
			x = rand.Gaus(conf_.x_offset, beam_size);
			y = rand.Gaus(conf_.y_offset, beam_size);
		} while (TMath::Abs(x - conf_.x_offset) > 3.0 * beam_size
				|| TMath::Abs(y - conf_.y_offset) > 3.0 * beam_size);

		double log_th = rand.Uniform(log_th_min, log_th_max);
		double theta = TMath::Exp(log_th);
		double phi = rand.Uniform(0, 2 * TMath::Pi());

		double log_thx = rand.Uniform(log_th_min, log_th_max);
		double log_thy = rand.Uniform(log_th_min, log_th_max);
		double log_xi = rand.Uniform(log_xi_min, log_xi_max);

		double thx = theta * TMath::Cos(phi) + conf_.x_half_crossing_angle;
		double thy = theta * TMath::Sin(phi) + conf_.y_half_crossing_angle;
		double xi = -TMath::Exp(log_xi);

		double in_prot[5];
		in_prot[0] = x;
		in_prot[1] = thx;
		in_prot[2] = y;
		in_prot[3] = thy;
		in_prot[4] = xi;

		ApertureTrackInfo ap_track_info;
		ap_track_info.in.SetValues(in_prot);

		int loss_count = 0;
		for (int i = 0; i < apertures_.size(); i++) {
			bool survived = apertures_[i].CheckAperture(in_prot);
			double out_pos[5];
			bool valid_out = apertures_[i].Transport(in_prot, out_pos, false);

			ApertureHitInfo hit_info;
			hit_info.out.SetValues(out_pos);
			hit_info.lost = !survived;
			hit_info.out_pos_valid = valid_out;
			ap_track_info.aperture_hits.push_back(hit_info);

			if (!survived)
				loss_count++;
		}

		//Store target tracks
		ApertureTrackInfo target_track_info;
		double out_pos[5];
		bool valid_out = parameterisation_.Transport(in_prot, out_pos, false);
		ApertureHitInfo target_hit_info;
		target_hit_info.out.SetValues(out_pos);
		target_hit_info.out_pos_valid = valid_out;
		target_track_info.aperture_hits.push_back(target_hit_info);

		bool track_lost_everywhere = (loss_count == apertures_.size());
		bool track_lost_nowhere = (loss_count == 0);

		total_track_counter++;
		if (total_track_counter % 200 == 0) {
			std::cout << "Total number of generated tracks: "
					<< total_track_counter << " Useful tracks:" << track_counter
					<< " out of " << conf_.lost_smaple_population << " needed"
					<< std::endl;
		}

		if (!track_lost_everywhere && !track_lost_nowhere) {
			track_counter++;
			ap_track_info.usefull_in_aperture_selection = true;
			aperture_tracks_.push_back(ap_track_info);
			target_tracks_.push_back(target_track_info);
		} else {
			ap_track_info.usefull_in_aperture_selection = false;
			aperture_tracks_.push_back(ap_track_info);
			target_tracks_.push_back(target_track_info);
		}
	}
}

void ApertureAnalyser::AnalyseApertureLosses(ApertureAnalysisConf &conf_) {
	bool lost_tracks_remain = true;
	std::set<int> identified_apertures;
	int loss_already_approximated = 0;
	out_file_ << "Aperture analysis results: " << std::endl;

	int usefull_track_number = 0;
	for (int i = 0; i < aperture_tracks_.size(); i++) {
		if (aperture_tracks_[i].usefull_in_aperture_selection) {
			usefull_track_number++;
		}
	}

	while (lost_tracks_remain) //iteration over apertures to find
	{
		std::vector<int> lost_particles_in_apertures;
		lost_particles_in_apertures.resize(apertures_.size());

		for (int i = 0; i < lost_particles_in_apertures.size(); i++)
			lost_particles_in_apertures[i] = 0;

		for (int i = 0; i < aperture_tracks_.size(); i++) //loop over lost tracks
				{
			std::set<int>::iterator used_apert_it =
					identified_apertures.begin();
			bool already_lost = false;
			for (; used_apert_it != identified_apertures.end(); ++used_apert_it) //check if the track was not lost in the already picked up apertures
					{
				already_lost =
						already_lost
								|| (aperture_tracks_[i].usefull_in_aperture_selection
										&& aperture_tracks_[i].aperture_hits[(*used_apert_it)].lost);
			}
			if (already_lost)
				continue;

			//not lost already
			for (int ap_id = 0; ap_id < lost_particles_in_apertures.size();
					++ap_id) {
				if (aperture_tracks_[i].usefull_in_aperture_selection
						&& aperture_tracks_[i].aperture_hits[ap_id].lost)
					lost_particles_in_apertures[ap_id]++;
			}
		}

		//find aperture with highest losses
		int max_loss = 0;
		int max_loss_id = -1;
		for (int i = 0; i < lost_particles_in_apertures.size(); i++) {
			if (lost_particles_in_apertures[i] > max_loss) {
				max_loss = lost_particles_in_apertures[i];
				max_loss_id = i;
			}
		}
		if (max_loss_id == -1) {
			//no more losses in the remaining apertures
			break;
		} else {
			loss_already_approximated += max_loss;
			std::cout << "max_loss_id=" << max_loss_id << " Loss_fraction="
					<< (double) max_loss / (double) usefull_track_number
					<< " Losses_already_approximated="
					<< (double) loss_already_approximated
							/ (double) usefull_track_number << " aperture_name="
					<< apertures_[max_loss_id].GetName() << std::endl;

			out_file_ << "max_loss_id=" << max_loss_id << " Loss_fraction="
					<< (double) max_loss / (double) usefull_track_number
					<< " Losses_already_approximated="
					<< (double) loss_already_approximated
							/ (double) usefull_track_number << " aperture_name="
					<< apertures_[max_loss_id].GetName() << std::endl;

			ApertureHistogramsEntity aper_losses(
					apertures_[max_loss_id].GetName(), conf_);

			char prefix[128];
			sprintf(prefix, "%02d_", max_loss_id + 1);
			aper_losses.AddNamePrefix(prefix);

			for (int i = 0; i < aperture_tracks_.size(); i++) //loop over lost tracks
					{
				aper_losses.FillReferenceHists(aperture_tracks_[i].in, conf_);
				if (aperture_tracks_[i].aperture_hits[max_loss_id].lost /*&& aperture_tracks_[i].aperture_hits[max_loss_id].out_pos_valid*/) {
					aper_losses.FillApertureHitInfoSingleAperture(
							aperture_tracks_[i].in, conf_);
					if (aperture_tracks_[i].aperture_hits[max_loss_id].out_pos_valid) {
						aper_losses.FillApertureHitPositionSingleAperture(
								aperture_tracks_[i].in,
								aperture_tracks_[i].aperture_hits[max_loss_id].out,
								conf_);
					}
				} else //the track was not lost in the given aperture
				{
					aper_losses.FillSingleApertureSurvived(
							aperture_tracks_[i].in,
							target_tracks_[i].aperture_hits[0].out,
							target_tracks_[i].aperture_hits[0].out_pos_valid,
							conf_);
				}

				std::set<int>::iterator used_apert_it =
						identified_apertures.begin();
				bool already_lost = false;
				for (; used_apert_it != identified_apertures.end();
						++used_apert_it) //check if the track was not lost in the already picked up apertures
						{
					already_lost =
							already_lost
									|| aperture_tracks_[i].aperture_hits[(*used_apert_it)].lost;
				}
				if (!already_lost) {
					if (aperture_tracks_[i].aperture_hits[max_loss_id].lost /*&& aperture_tracks_[i].aperture_hits[max_loss_id].out_pos_valid*/) {
						aper_losses.FillApertureHitInfoSelectedSequence(
								aperture_tracks_[i].in, conf_);
						if (aperture_tracks_[i].aperture_hits[max_loss_id].out_pos_valid) {
							aper_losses.FillApertureHitPositionSelectedSequence(
									aperture_tracks_[i].in,
									aperture_tracks_[i].aperture_hits[max_loss_id].out,
									conf_);
						}
					} else //track survived the selected aperture sequence
					{
						aper_losses.FillTrackSurvivedSelectedApertures(
								aperture_tracks_[i].in,
								target_tracks_[i].aperture_hits[0].out,
								target_tracks_[i].aperture_hits[0].out_pos_valid,
								conf_);
					}
				}
				if (already_lost
						|| aperture_tracks_[i].aperture_hits[max_loss_id].lost) //the track lost in the apertures so far
						{
					aper_losses.FillTrackLostInSelectedApertures(
							aperture_tracks_[i].in,
							target_tracks_[i].aperture_hits[0].out,
							target_tracks_[i].aperture_hits[0].out_pos_valid,
							conf_);
				}
			}
			aperture_hists_.push_back(aper_losses);
			identified_apertures.insert(max_loss_id);
		}
	}
}

void ApertureAnalyser::AnalyseAllApertureAcceptances(
		ApertureAnalysisConf &conf_) {
	for (int i = 0; i < apertures_.size(); i++) {
		std::string name = "all_aper_";
		name += apertures_[i].GetName();
		all_aperture_hists_.push_back(ApertureHistogramsEntity(name, conf_));
	}

	std::vector<int> losses_in_apertures;
	losses_in_apertures.resize(apertures_.size());
	for (int i = 0; i < losses_in_apertures.size(); i++) {
		losses_in_apertures[i] = 0;
	}

	for (int i = 0; i < aperture_tracks_.size(); i++) //loop over lost tracks
			{
		bool track_lost = false;
		for (int ap_id = 0; ap_id < aperture_tracks_[i].aperture_hits.size();
				++ap_id) //loop over apertures
				{
			if (!track_lost) {
				if (aperture_tracks_[i].aperture_hits[ap_id].lost /*&& aperture_tracks_[i].aperture_hits[ap_id].out_pos_valid*/) {
					losses_in_apertures[ap_id]++;
					all_aperture_hists_[ap_id].FillApertureHitInfoSelectedSequence(
							aperture_tracks_[i].in, conf_);
					if (aperture_tracks_[i].aperture_hits[ap_id].out_pos_valid) {
						all_aperture_hists_[ap_id].FillApertureHitPositionSelectedSequence(
								aperture_tracks_[i].in,
								aperture_tracks_[i].aperture_hits[ap_id].out,
								conf_);
					}
				} else //track not lost so far
				{
					all_aperture_hists_[ap_id].FillTrackSurvivedSelectedApertures(
							aperture_tracks_[i].in,
							target_tracks_[i].aperture_hits[0].out,
							target_tracks_[i].aperture_hits[0].out_pos_valid,
							conf_);
				}
			}
			if (aperture_tracks_[i].aperture_hits[ap_id].lost /*&& aperture_tracks_[i].aperture_hits[ap_id].out_pos_valid*/) {
				all_aperture_hists_[ap_id].FillApertureHitInfoSingleAperture(
						aperture_tracks_[i].in, conf_);
				if (aperture_tracks_[i].aperture_hits[ap_id].out_pos_valid) {
					all_aperture_hists_[ap_id].FillApertureHitPositionSingleAperture(
							aperture_tracks_[i].in,
							aperture_tracks_[i].aperture_hits[ap_id].out,
							conf_);
				}
			} else //track survived single aperture
			{
				all_aperture_hists_[ap_id].FillSingleApertureSurvived(
						aperture_tracks_[i].in,
						target_tracks_[i].aperture_hits[0].out,
						target_tracks_[i].aperture_hits[0].out_pos_valid,
						conf_);
			}

			all_aperture_hists_[ap_id].FillReferenceHists(
					aperture_tracks_[i].in, conf_);

			track_lost = track_lost
					|| aperture_tracks_[i].aperture_hits[ap_id].lost;
			if (track_lost) //track lost in sequence so far
			{
				all_aperture_hists_[ap_id].FillTrackLostInSelectedApertures(
						aperture_tracks_[i].in,
						target_tracks_[i].aperture_hits[0].out,
						target_tracks_[i].aperture_hits[0].out_pos_valid,
						conf_);
			}
		}
	}

	out_file_ << std::endl << std::endl << "Real losses in apertures:"
			<< std::endl;
	std::multimap<int, std::pair<int, std::string> > sorted_losses;
	for (int i = 0; i < losses_in_apertures.size(); i++) {
		sorted_losses.insert(
				std::pair<int, std::pair<int, std::string> >(
						losses_in_apertures[i],
						std::pair<int, std::string>(i,
								apertures_[i].GetName())));
		out_file_ << "Aperture name=" << apertures_[i].GetName() << " id="
				<< i + 1 << " lost particles=" << losses_in_apertures[i]
				<< " fraction="
				<< (double) losses_in_apertures[i] / aperture_tracks_.size()
				<< std::endl;
	}

	std::multimap<int, std::pair<int, std::string> >::reverse_iterator it;
	out_file_ << std::endl << std::endl << "Real losses in apertures, sorted:"
			<< std::endl;
	for (it = sorted_losses.rbegin(); it != sorted_losses.rend(); ++it) {
		out_file_ << "Aperture name=" << it->second.second << " id="
				<< it->second.first + 1 << " lost particles=" << it->first
				<< " fraction=" << (double) it->first / aperture_tracks_.size()
				<< std::endl;
	}
}

void ApertureAnalyser::AnalyseDestinationAcceptance(
		ApertureAnalysisConf &conf_) {
}

void ApertureAnalyser::FindTheBottleneckApertures(ApertureAnalysisConf &conf_) {
	std::vector<int> lost_particles_in_apertures;
	lost_particles_in_apertures.resize(apertures_.size());

	for (int i = 0; i < lost_particles_in_apertures.size(); i++)
		lost_particles_in_apertures[i] = 0;

	int all_usefull_particles = 0;
	for (int i = 0; i < aperture_tracks_.size(); i++) //loop over lost tracks
			{
		if (aperture_tracks_[i].usefull_in_aperture_selection) {
			all_usefull_particles++;
		}

		for (int ap_id = 0; ap_id < aperture_tracks_[i].aperture_hits.size();
				++ap_id) //loop over apertures
				{
			if (aperture_tracks_[i].usefull_in_aperture_selection
					&& aperture_tracks_[i].aperture_hits[ap_id].lost) {
				lost_particles_in_apertures[ap_id]++;
			}
		}
	}

	std::multimap<int, std::pair<int, std::string> > sorted_losses;

	for (int i = 0; i < lost_particles_in_apertures.size(); i++) {
		sorted_losses.insert(
				std::pair<int, std::pair<int, std::string> >(
						lost_particles_in_apertures[i],
						std::pair<int, std::string>(i,
								apertures_[i].GetName())));
	}

	out_file_ << std::endl << std::endl
			<< "The sequence of the bottleneck apertures:" << std::endl;
	std::multimap<int, std::pair<int, std::string> >::reverse_iterator it;
	for (it = sorted_losses.rbegin(); it != sorted_losses.rend(); ++it) {
		out_file_ << "Aperture name=" << it->second.second << " id="
				<< it->second.first + 1 << " lost particles=" << it->first
				<< " fraction=" << (double) it->first / all_usefull_particles
				<< std::endl;
	}
}

void ApertureAnalyser::OpenInfoTextFile(ApertureAnalysisConf &conf_) {
	out_file_.open(conf_.analysis_output_file.c_str(), std::ios::out);
}

void ApertureAnalyser::CloseInfoTextFile(ApertureAnalysisConf &conf_) {
	out_file_.close();
}

void ApertureAnalyser::WriteApertureHistograms(ApertureAnalysisConf &conf_) {
	TFile *f = TFile::Open(conf_.analysis_output_hist_file.c_str(), "recreate");
	f->mkdir("selected_apertures");
	f->cd("selected_apertures");

	for (int i = 0; i < aperture_hists_.size(); i++) {
		char prefix[128];
		sprintf(prefix, "%02d_", i + 1);
		aperture_hists_[i].AddNamePrefix(prefix);
		aperture_hists_[i].Write();
	}
	f->cd("..");

	f->mkdir("all_apertures");
	f->cd("all_apertures");

	for (int i = 0; i < all_aperture_hists_.size(); i++) {
		char prefix[128];
		sprintf(prefix, "%02d_", i + 1);
		all_aperture_hists_[i].AddNamePrefix(prefix);
		all_aperture_hists_[i].Write();
	}

	f->cd("..");
	f->Close();
}

