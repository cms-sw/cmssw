#include "MADParamGenerator.h"
#include "TTimeStamp.h"
#include "TRandom3.h"
#include "TNtupleD.h"
#include "TNtupleDcorr.h"

#include "FitData.h"
#include <boost/shared_ptr.hpp>
#include "CurrentMemoryUsage.h"
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

void MADParamGenerator::GenerateMADConfFile(const std::string &base_conf_file,
		const std::string &out_conf_file, const std::string &from_marker_name,
		double from_marker_s_pos, bool define_from,
		const std::string &to_marker_name, double to_marker_s_pos,
		bool define_to, int particles_number, bool aperture_limit,
		std::vector<std::string> scoring_planes, const std::string &beam) {
	std::fstream input;
	input.open(base_conf_file.c_str(), std::ios::in);

	std::fstream output;
	output.open(out_conf_file.c_str(), std::ios::out);

	Conf_file_processing(input, output, from_marker_name, from_marker_s_pos,
			define_from, to_marker_name, to_marker_s_pos, define_to,
			particles_number, aperture_limit, scoring_planes, beam);
	input.close();
}

void MADParamGenerator::Conf_file_processing(std::fstream &base_conf_file,
		std::fstream & conf_file, const std::string &from_marker_name,
		double from_marker_s_pos, bool define_from,
		const std::string &to_marker_name, double to_marker_s_pos,
		bool define_to, int particles_number, bool aperture_limit,
		const std::vector<std::string> &scoring_planes,
		const std::string &beam) {
	while (base_conf_file.good() && !base_conf_file.eof()) {
		char curr_char;
		base_conf_file.get(curr_char);

		if (curr_char == '#') {
			std::string token = GetToken(base_conf_file);
			ProcessToken(conf_file, from_marker_name, from_marker_s_pos,
					define_from, to_marker_name, to_marker_s_pos, define_to,
					particles_number, aperture_limit, token, scoring_planes,
					beam);
		} else {
			conf_file << curr_char;
		}
	}
}

std::string MADParamGenerator::GetToken(std::fstream &base_conf_file) {
	char character;

	char temp_token[512];
	base_conf_file.get(temp_token, 512, '#');
	if (base_conf_file.good() && !base_conf_file.eof())
		base_conf_file.get(character);

	return std::string(temp_token);
}

void MADParamGenerator::ProcessToken(std::fstream &conf_file,
		const std::string &from_marker_name, double from_marker_s_pos,
		bool define_from, const std::string & to_marker_name,
		double to_marker_s_pos, bool define_to, int particles_number,
		bool aperture_limit, const std::string &token,
		const std::vector<std::string> &scoring_planes,
		const std::string &beam) {
	std::string ptcPrefix = "ptc_";
	if (aperture_limit)
		ptcPrefix = "";
	if (token == "header_placement") {
		conf_file << "! defines a macro to read initial coordinates"
				<< std::endl;
		conf_file << "getpart(nx): macro = {" << std::endl;
		conf_file << " ex   = table(myevent,nx,trx);" << std::endl;
		conf_file << " epx   = table(myevent,nx,trpx);" << std::endl;
		conf_file << " ey   = table(myevent,nx,try);" << std::endl;
		conf_file << " epy   = table(myevent,nx,trpy);" << std::endl;
		conf_file << " et    = table(myevent,nx,tt);" << std::endl;
		conf_file << " ept    = table(myevent,nx,tpt);" << std::endl;
		conf_file << " value,ex,epx,ey,epy,et,ept;" << std::endl;
		conf_file << "}" << std::endl;
	} else if (token == "scoring_plane_definition") {
		//beginning marker definition
		if (define_from) {
			conf_file << "" << from_marker_name << " : marker;" << std::endl;
			conf_file << "seqedit,sequence=" << beam << ";" << std::endl;
			conf_file << "install,element=" << from_marker_name << ",at="
					<< from_marker_s_pos << ",from=ip5;" << std::endl;
			conf_file << "endedit;" << std::endl;
		}

		//end marker definition
		if (define_to) {
			conf_file << "" << to_marker_name << " : marker;" << std::endl;
			conf_file << "seqedit,sequence=" << beam << ";" << std::endl;
			conf_file << "install,element=" << to_marker_name << ",at="
					<< to_marker_s_pos << ",from=ip5;" << std::endl;
			conf_file << "endedit;" << std::endl;
		}
	} else if (token == "start_point") {
		conf_file << from_marker_name;
	} else if (token == "end_point") {
		conf_file << to_marker_name;
	} else if (token == "scoring_plane_placement") {
		conf_file << ptcPrefix << "observe,place=" << to_marker_name << ";"
				<< std::endl;
		for (int i = 0; i < scoring_planes.size(); i++) {
			conf_file << ptcPrefix << "observe,place=" << scoring_planes[i]
					<< ";" << std::endl;
		}
	} else if (token == "import_particles") {
		conf_file << "readmytable,file=part.in,table=myevent;" << std::endl;
	} else if (token == "insert_particles") {
		conf_file << "! read in initial coordinates at set for tracking"
				<< std::endl;
		conf_file << "   n=1;" << std::endl;
		conf_file << "   while ( n < " << particles_number << " + 1 ) {"
				<< std::endl;
		conf_file << "      exec,getpart($n);" << std::endl;
		conf_file << "      " << ptcPrefix
				<< "start,x=ex,px=epx,y=ey,py=epy,t=et,pt=ept;" << std::endl;
		conf_file << "      n = n + 1;" << std::endl;
		conf_file << "   }" << std::endl;
	} else if (token == "output_mad_file") {
		conf_file << "track";
	} else if (token == "options") {
		if (aperture_limit)
			conf_file << ",aperture";
		if (aperture_limit)
			conf_file << ",recloss";
	} else if (token == "save_lost_particles") {
		if (aperture_limit)
			conf_file << "write,table=trackloss,file=\"trackloss\""
					<< std::endl;
	} else if (token == "beam_type") {
		if (beam == "lhcb1" || beam == "lhcb2")
			conf_file << beam;
		else
			conf_file << "lhcb1";
	} else if (token == "beam_bv") {
		if (beam == "lhcb1")
			conf_file << "+1";
		else if (beam == "lhcb2")
			conf_file << "-1";
		else
			conf_file << "+1";
	}
}

void MADParamGenerator::GenerateRandomSamples(int number_of_particles,
		double x_min, double x_max, double theta_x_min, double theta_x_max,
		double y_min, double y_max, double theta_y_min, double theta_y_max,
		double ksi_min, double ksi_max, const std::string &out_file_name) {
	std::cout << "generate random samples: start\n";
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	for (i = 0; i < number_of_particles; i++) {
		x = r.Uniform(x_min, x_max);
		theta_x = r.Uniform(theta_x_min, theta_x_max);
		y = r.Uniform(y_min, y_max);
		theta_y = r.Uniform(theta_y_min, theta_y_max);
		ksi = r.Uniform(ksi_min, ksi_max);
		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
	std::cout << "generate random samples: end\n";
}

void MADParamGenerator::GenerateDebugRandomSamples(int number_of_particles,
		double x_min, double x_max, double theta_x_min, double theta_x_max,
		double y_min, double y_max, double theta_y_min, double theta_y_max,
		double ksi_min, double ksi_max, const std::string &out_file_name) {
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	int ksi_points_no = 11;
	int t_points_no = 3;
	//  int v_points_no = 3;

	double vx_mean = (x_min + x_max) / 2.0;
	double vy_mean = (y_min + y_max) / 2.0;
	double vx_sigma = (x_max - vx_mean);
	double vy_sigma = (y_max - vy_mean);

	double tx_mean = (theta_x_min + theta_x_max) / 2.0;
	double ty_mean = (theta_y_min + theta_y_max) / 2.0;
	double tx_sigma = (theta_x_max - tx_mean);
	double ty_sigma = (theta_y_max - ty_mean);

	for (i = 0; i < number_of_particles; i++) {
		unsigned int ksi_ind = r.Integer(ksi_points_no);
		unsigned int t_ind = r.Integer(t_points_no);
		//    unsigned int v_ind = r.Integer(v_points_no);

		//    double t = 7e3*7e3*tx_sigma*tx_sigma;
		double theta_angle = tx_sigma / TMath::Power(10, t_ind / 2.0);
		double angle_phi = r.Uniform(0.0, 2 * TMath::Pi());
		double theta_x = tx_mean + theta_angle * TMath::Cos(angle_phi);
		double theta_y = ty_mean + theta_angle * TMath::Sin(angle_phi);
		//    std::cout<<tx_sigma<<" "<<t_ind<<" "<<theta_angle<<" "<<angle_phi<<" "<<theta_x<<" "<<theta_y<<std::endl;

		double phi_v = r.Uniform(0.0, 2 * TMath::Pi());
		double vr = r.Gaus(0.0, vx_sigma);
		double x = vx_mean + vr * TMath::Cos(phi_v);
		double y = vy_mean + vr * TMath::Sin(phi_v);

		double ksi = ksi_min
				+ (ksi_max - ksi_min) * (double) ksi_ind
						/ ((double) ksi_points_no - 1.0);

		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
}

void MADParamGenerator::GenerateGridDebugRandomSamples(int number_of_particles,
		double x_min, double x_max, double theta_x_min, double theta_x_max,
		double y_min, double y_max, double theta_y_min, double theta_y_max,
		double ksi_min, double ksi_max, const std::string &out_file_name) {
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	int ksi_points_no = 11;
	int t_points_no = 3;
	int v_points_no = 3;

	double vx_mean = (x_min + x_max) / 2.0;
	double vy_mean = (y_min + y_max) / 2.0;
	double vx_spread = (x_max - vx_mean);
	double vy_spread = (y_max - vy_mean);

	double tx_mean = (theta_x_min + theta_x_max) / 2.0;
	double ty_mean = (theta_y_min + theta_y_max) / 2.0;
	double tx_spread = (theta_x_max - tx_mean);
	double ty_spread = (theta_y_max - ty_mean);

	for (i = 0; i < number_of_particles; i++) {
		int ksi_ind = r.Integer(ksi_points_no);
		int tx_ind = r.Integer(2 * t_points_no + 1) - t_points_no;
		int ty_ind = r.Integer(2 * t_points_no + 1) - t_points_no;
		int vx_int = r.Integer(2 * v_points_no + 1) - v_points_no;
		int vy_int = r.Integer(2 * v_points_no + 1) - v_points_no;

		double cont_ksi = r.Uniform(ksi_min, ksi_max);
		double cont_x = r.Uniform(x_min, x_max);
		double cont_theta_x = r.Uniform(theta_x_min, theta_x_max);
		double cont_y = r.Uniform(y_min, y_max);
		double cont_theta_y = r.Uniform(theta_y_min, theta_y_max);

		int mode = r.Integer(5); //x, y, thetax, thetay, ksi

		double x, y, theta_x, theta_y, ksi;
		double dist_x, dist_y, dist_theta_x, dist_theta_y, dist_ksi;

		dist_ksi = ksi_min
				+ (ksi_max - ksi_min) * (double) ksi_ind
						/ ((double) ksi_points_no - 1.0);

		if (tx_ind != 0)
			dist_theta_x = TMath::Sign(1, tx_ind) * tx_spread
					/ TMath::Power(10, (TMath::Abs(tx_ind) - 1) / 2.0)
					+ tx_mean;
		else
			dist_theta_x = tx_mean;

		if (ty_ind != 0)
			dist_theta_y = TMath::Sign(1, ty_ind) * ty_spread
					/ TMath::Power(10, (TMath::Abs(ty_ind) - 1) / 2.0)
					+ ty_mean;
		else
			dist_theta_y = ty_mean;

		dist_x = vx_mean + vx_spread * vx_int / (v_points_no);
		dist_y = vy_mean + vy_spread * vy_int / (v_points_no);

		switch (mode) {
		case 0: //ksi continuous
			ksi = dist_ksi;
			theta_x = dist_theta_x;
			theta_y = dist_theta_y;
			x = dist_x;
			y = dist_y;
			break;
		case 1: //thetax continuous
			ksi = dist_ksi;
			theta_x = cont_theta_x;
			theta_y = dist_theta_y;
			x = dist_x;
			y = dist_y;
			break;
		case 2: //thetay continuous
			ksi = dist_ksi;
			theta_x = dist_theta_x;
			theta_y = cont_theta_y;
			x = dist_x;
			y = dist_y;
			break;
		case 3: //vx continuous
			ksi = dist_ksi;
			theta_x = dist_theta_x;
			theta_y = dist_theta_y;
			x = cont_x;
			y = dist_y;
			break;
		case 4: //vy continuous
			ksi = dist_ksi;
			theta_x = dist_theta_x;
			theta_y = dist_theta_y;
			x = dist_x;
			y = cont_y;
			break;
		}

		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
}

void MADParamGenerator::GenerateXiContTDiscPhiContDebugRandomSamples(
		int number_of_particles, double x_min, double x_max, double theta_x_min,
		double theta_x_max, double y_min, double y_max, double theta_y_min,
		double theta_y_max, double ksi_min, double ksi_max,
		const std::string &out_file_name) {
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	int ksi_points_no = 11;
	int t_points_no = 1;
	//  int v_points_no = 3;

	double vx_mean = (x_min + x_max) / 2.0;
	double vy_mean = (y_min + y_max) / 2.0;
	double vx_sigma = (x_max - vx_mean);
	double vy_sigma = (y_max - vy_mean);

	double tx_mean = (theta_x_min + theta_x_max) / 2.0;
	double ty_mean = (theta_y_min + theta_y_max) / 2.0;
	double tx_sigma = (theta_x_max - tx_mean);
	double ty_sigma = (theta_y_max - ty_mean);

	for (i = 0; i < number_of_particles; i++) {
		unsigned int ksi_ind = r.Integer(ksi_points_no);
		unsigned int t_ind = r.Integer(t_points_no);
		//    unsigned int v_ind = r.Integer(v_points_no);

		//    double t = 7e3*7e3*tx_sigma*tx_sigma;
		double theta_angle = tx_sigma / TMath::Power(10, t_ind / 2.0);
		double angle_phi = r.Uniform(0.0, 2 * TMath::Pi());
		double theta_x = tx_mean + theta_angle * TMath::Cos(angle_phi);
		double theta_y = ty_mean + theta_angle * TMath::Sin(angle_phi);
		//    std::cout<<tx_sigma<<" "<<t_ind<<" "<<theta_angle<<" "<<angle_phi<<" "<<theta_x<<" "<<theta_y<<std::endl;

		double phi_v = r.Uniform(0.0, 2 * TMath::Pi());
		double vr = r.Gaus(0.0, vx_sigma);
		double x = vx_mean + vr * TMath::Cos(phi_v);
		double y = vy_mean + vr * TMath::Sin(phi_v);

		double ksi = r.Uniform(ksi_min, ksi_max);

		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
}

bool MADParamGenerator::ComputeTheta(double beam_energy, double ksi, double t,
		double &theta) {
	double m0_ = 0.93827201323;
	double p1_ = TMath::Sqrt(beam_energy * beam_energy - m0_ * m0_);

	double term1 = 1 + ksi;
	double term2 = term1 * term1;
	double term3 = m0_ * m0_;
	double term4 = p1_ * p1_;
	double term5 = term3 * term3;
	double term6 = ksi * ksi;
	double sqrt1 = TMath::Sqrt(term3 + term4);
	double sqrt2 = TMath::Sqrt(term3 + term2 * term4);
	double denom1 = term2 * term4 * term4;
	double denom2 = term2 * term4;

	double bracket1 = -2 * term5 / denom1 - 2 * term3 / denom2
			- 2 * ksi * term3 / denom2 - term6 * term3 / denom2
			+ 2 * term3 * sqrt1 * sqrt2 / denom1;

	double bracket2 = t
			* (term3 / denom1 - t / (4 * denom1) - sqrt1 * sqrt2 / denom1);

	double theta_squared = bracket1 + bracket2;

	bool res = false;
	theta = 0.0;

	if (theta_squared >= 0.0 && theta_squared < 0.01) {
		theta = TMath::Sqrt(theta_squared);
		res = true;
	}

	res = res || (ksi == 0.0 && t == 0.0);
	return res;
}

void MADParamGenerator::GenerateDiffractiveProtonsSamples(
		int number_of_particles, double beam_energy, double x_min, double x_max,
		double theta_x_min, double theta_x_max, double y_min, double y_max,
		double theta_y_min, double theta_y_max, double ksi_min, double ksi_max,
		const std::string &out_file_name) {
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	int ksi_points_no = 11;
	int t_points_no = 1;
	//  int v_points_no = 3;

	double vx_mean = (x_min + x_max) / 2.0;
	double vy_mean = (y_min + y_max) / 2.0;
	double vx_sigma = (x_max - vx_mean);
	double vy_sigma = (y_max - vy_mean);

	double tx_mean = (theta_x_min + theta_x_max) / 2.0;
	double ty_mean = (theta_y_min + theta_y_max) / 2.0;
	double tx_sigma = (theta_x_max - tx_mean);
	double ty_sigma = (theta_y_max - ty_mean);

	double t_min = 1e-3;
	double t_max = 100;

	double ksi_min_ = 1e-7;
	double ksi_max_ = 3e-1;
	double B = 5;

	double x_xi_min = TMath::Log(ksi_min_);
	double x_xi_max = TMath::Log(ksi_max_);
	//double x_t_min = TMath::Exp(-B*t_min);
	//double x_t_max = TMath::Exp(-B*t_max);
	double x_t_min = TMath::Log(t_min);
	double x_t_max = TMath::Log(t_max);

	for (i = 0; i < number_of_particles; i++) {
		double ksi = 0.;
		double t = 0.;
		double theta_angle = 0.;
		bool angle_gen_ok = false;
		do {
			double x_xi = r.Uniform(x_xi_min, x_xi_max);
			ksi = -TMath::Exp(x_xi);

			double x_t = r.Uniform(x_t_min, x_t_max);
			//double t = -TMath::Log(x_t)/B;
			t = -TMath::Exp(x_t);
			//theta_angle = TMath::Sqrt(t)/beam_energy;
			angle_gen_ok = ComputeTheta(beam_energy, ksi, t, theta_angle);
		} while (!angle_gen_ok);

		double angle_phi = r.Uniform(0.0, 2 * TMath::Pi());

		double theta_x = tx_mean + theta_angle * TMath::Cos(angle_phi);
		double theta_y = ty_mean + theta_angle * TMath::Sin(angle_phi);
		double x = vx_mean + r.Gaus(0.0, vx_sigma / TMath::Sqrt(2.0));
		double y = vy_mean + r.Gaus(0.0, vy_sigma / TMath::Sqrt(2.0));

		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
}

void MADParamGenerator::GenerateElasticProtonsSamples(int number_of_particles,
		double beam_energy, double x_min, double x_max, double theta_x_min,
		double theta_x_max, double y_min, double y_max, double theta_y_min,
		double theta_y_max, double ksi_min, double ksi_max,
		const std::string &out_file_name) {
	std::ofstream ofs(out_file_name.c_str());

	double x, theta_x, y, theta_y, ksi;

	TTimeStamp time;

	TRandom3 r(time.GetSec() + time.GetNanoSec());

	Int_t i;

	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;

	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt"
			<< std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le"
			<< std::endl;

	int ksi_points_no = 11;
	int t_points_no = 1;
	//  int v_points_no = 3;

	double vx_mean = (x_min + x_max) / 2.0;
	double vy_mean = (y_min + y_max) / 2.0;
	double vx_sigma = (x_max - vx_mean);
	double vy_sigma = (y_max - vy_mean);

	double tx_mean = (theta_x_min + theta_x_max) / 2.0;
	double ty_mean = (theta_y_min + theta_y_max) / 2.0;
	double tx_sigma = (theta_x_max - tx_mean);
	double ty_sigma = (theta_y_max - ty_mean);

	double t_min = 1e-4;
	double t_max = 100;

	double ksi_min_ = 0;
	double ksi_max_ = 0;
	double B = 5;

	double x_xi_min = TMath::Log(ksi_min_);
	double x_xi_max = TMath::Log(ksi_max_);
	double x_t_min = TMath::Log(t_min);
	double x_t_max = TMath::Log(t_max);

	for (i = 0; i < number_of_particles; i++) {
		double ksi = 0.;
		double t = 0.;
		double theta_angle = 0.;
		bool angle_gen_ok = false;
		ksi = 0;
		double x_t = r.Uniform(x_t_min, x_t_max);
		t = -TMath::Exp(x_t);
		theta_angle = TMath::Sqrt(-t) / beam_energy;

		double angle_phi = r.Uniform(0.0, 2 * TMath::Pi());

		double theta_x = tx_mean + theta_angle * TMath::Cos(angle_phi);
		double theta_y = ty_mean + theta_angle * TMath::Sin(angle_phi);
		double x = vx_mean + r.Gaus(0.0, vx_sigma / TMath::Sqrt(2.0));
		double y = vy_mean + r.Gaus(0.0, vy_sigma / TMath::Sqrt(2.0));

		ofs.precision(25);

		ofs << "    \"" << i + 1 << "\" " << x << " " << theta_x << " " << y
				<< " " << theta_y << " 0.0 " << ksi << " " << std::endl;
	}
	ofs.close();
}

void MADParamGenerator::RunMAD(const std::string &conf_file) {
	std::cout << "runmad: start\n";
	std::string cmd;
	cmd = "rm -f ./trackloss";
	system(cmd.c_str());
	cmd = std::string("madx < ") + conf_file + " >/dev/null";
	system(cmd.c_str());
	std::cout << "runmad: end\n";
}

int MADParamGenerator::AppendRootTree(std::string root_file_name,
		std::string out_prefix, std::string out_station, bool recloss,
		std::string lost_particles_tree_filename,
		const std::vector<std::string> &scoring_planes, bool compare_apert) {
	std::cout << "append root tree: start\n";
	FitData text2rootconverter;
	text2rootconverter.readIn("part.in");
	text2rootconverter.readOut("trackone", out_station.c_str());
	text2rootconverter.readAdditionalScoringPlanes("trackone", scoring_planes);

	int added_entries = 0;

	if (!recloss || compare_apert) {
		TFile *f = TFile::Open(root_file_name.c_str(), "UPDATE");
		TTree *tree = (TTree*) f->Get("transport_samples");
		if (!tree) {
			tree = CreateSamplesTree(f, out_prefix, scoring_planes);
		}
		added_entries = text2rootconverter.AppendRootFile(tree, out_prefix);
		f->cd();
		tree->SetBranchStatus("*", 1); //enable all branches
		tree->Write(NULL, TObject::kOverwrite);

		if (compare_apert) {
			TTree *acc_acept_tree = (TTree*) f->Get("acc_acept_tree");
			if (!acc_acept_tree) {
				acc_acept_tree = CreateAccelAcceptTree(f);
			}
			text2rootconverter.AppendAcceleratorAcceptanceRootFile(
					acc_acept_tree);
			f->cd();
			acc_acept_tree->SetBranchStatus("*", 1); //enable all branches
			acc_acept_tree->Write(NULL, TObject::kOverwrite);
		}

		f->Close();
		delete f;

	}

	if (recloss && !compare_apert) {
		TFile *lost_particles_file = TFile::Open(
				lost_particles_tree_filename.c_str(), "UPDATE");
		TTree *lost_particles_tree = (TTree*) lost_particles_file->Get(
				"lost_particles");
		if (!lost_particles_tree) {
			lost_particles_tree = CreateLostParticlesTree(lost_particles_file);
			lost_particles_tree->Print();
		}
		text2rootconverter.readLost("trackloss");
		added_entries = text2rootconverter.AppendLostParticlesRootFile(
				lost_particles_tree);
		lost_particles_tree->Write(NULL, TObject::kOverwrite);
		lost_particles_file->Close();
		delete lost_particles_file;
	}
	std::cout << "append root tree: end\n";
	return added_entries;
}

Long64_t MADParamGenerator::GetNumberOfEntries(std::string root_file_name,
		std::string out_prefix) {
	std::cout << "get number of entries: start\n";
	TFile *f = TFile::Open(root_file_name.c_str(), "read");
	if (!f->IsOpen())
		return 0;

	TTree *tree = (TTree*) f->Get("transport_samples");
	if (!tree)
		return 0;

	Long64_t entries = tree->GetEntries();
	f->Close();
	delete f;
	std::cout << "get number of entries: end\n";
	return entries;
}

Long64_t MADParamGenerator::GetLostParticlesEntries(
		const Parametisation_configuration &conf) {
	TFile *f = TFile::Open(conf.lost_particles_tree_filename.c_str(), "read");
	if (!f->IsOpen())
		return 0;

	TTree *tree = (TTree*) f->Get("lost_particles");
	if (!tree)
		return 0;

	Long64_t entries = tree->GetEntries();
	f->Close();
	delete f;
	return entries;
}

TTree *MADParamGenerator::CreateSamplesTree(TFile *f, std::string out_prefix,
		const std::vector<std::string> &scoring_planes) {
	std::string varlist;

	std::string x_in_lab = "x_in";
	std::string theta_x_in_lab = "theta_x_in";
	std::string y_in_lab = "y_in";
	std::string theta_y_in_lab = "theta_y_in";
	std::string ksi_in_lab = "ksi_in";
	std::string s_in_lab = "s_in";

	std::string x_out_lab = out_prefix + "_x_out";
	std::string theta_x_out_lab = out_prefix + "_theta_x_out";
	std::string y_out_lab = out_prefix + "_y_out";
	std::string theta_y_out_lab = out_prefix + "_theta_y_out";
	std::string ksi_out_lab = out_prefix + "_ksi_out";
	std::string s_out_lab = out_prefix + "_s_out";
	std::string valid_out_lab = out_prefix + "_valid_out";

	varlist = x_in_lab + ":" + theta_x_in_lab + ":" + y_in_lab + ":"
			+ theta_y_in_lab + ":" + ksi_in_lab + ":" + s_in_lab + ":"
			+ x_out_lab + ":" + theta_x_out_lab + ":" + y_out_lab + ":"
			+ theta_y_out_lab + ":" + ksi_out_lab + ":" + s_out_lab + ":"
			+ valid_out_lab;

	for (int i = 0; i < scoring_planes.size(); i++) {
		varlist += std::string(":") + scoring_planes[i] + "_x_out";
		varlist += std::string(":") + scoring_planes[i] + "_theta_x_out";
		varlist += std::string(":") + scoring_planes[i] + "_y_out";
		varlist += std::string(":") + scoring_planes[i] + "_theta_y_out";
		varlist += std::string(":") + scoring_planes[i] + "_ksi_out";
		varlist += std::string(":") + scoring_planes[i] + "_s_out";
		varlist += std::string(":") + scoring_planes[i] + "_valid_out";
	}

	f->cd();
	return new TNtupleDcorr("transport_samples", "transport_samples",
			varlist.c_str());
}

TTree *MADParamGenerator::CreateAccelAcceptTree(TFile *f, std::string name) {
	std::string varlist = "x:theta_x:y:theta_y:ksi:mad_accept:par_accept";
	f->cd();
	return new TNtupleDcorr(name.c_str(), name.c_str(), varlist.c_str());
}

TTree *MADParamGenerator::CreateLostParticlesTree(TFile *lost_particles_file) {
	std::string varlist;

	std::string br_x_in_lab = "in_x";
	std::string br_theta_x_in_lab = "in_theta_x";
	std::string br_y_in_lab = "in_y";
	std::string br_theta_y_in_lab = "in_theta_y";
	std::string br_ksi_in_lab = "in_ksi";
	std::string br_s_in_lab = "in_s";

	std::string br_x_out_lab = "out_x";
	std::string br_theta_x_out_lab = "out_theta_x";
	std::string br_y_out_lab = "out_y";
	std::string br_theta_y_out_lab = "out_theta_y";
	std::string br_ksi_out_lab = "out_ksi";
	std::string br_s_out_lab = "out_s";

	std::string br_element_out_lab = "out_element";

	std::string x_in_lab = br_x_in_lab + "/D";
	std::string theta_x_in_lab = br_theta_x_in_lab + "/D";
	std::string y_in_lab = br_y_in_lab + "/D";
	std::string theta_y_in_lab = br_theta_y_in_lab + "/D";
	std::string ksi_in_lab = br_ksi_in_lab + "/D";
	std::string s_in_lab = br_s_in_lab + "/D";

	std::string x_out_lab = br_x_out_lab + "/D";
	std::string theta_x_out_lab = br_theta_x_out_lab + "/D";
	std::string y_out_lab = br_y_out_lab + "/D";
	std::string theta_y_out_lab = br_theta_y_out_lab + "/D";
	std::string ksi_out_lab = br_ksi_out_lab + "/D";
	std::string s_out_lab = br_s_out_lab + "/D";

	std::string element_out_lab = br_element_out_lab + "/C";

	double x_in, theta_x_in, y_in, theta_y_in, ksi_in, s_in;
	double x_out, theta_x_out, y_out, theta_y_out, ksi_out, s_out;
	char element_out[512];
	element_out[0] = 0;

	lost_particles_file->cd();
	TTree *lost_part_tree = new TTree("lost_particles", "lost_particles");

	lost_part_tree->Branch(br_x_in_lab.c_str(), &x_in, x_in_lab.c_str());
	lost_part_tree->Branch(br_theta_x_in_lab.c_str(), &theta_x_in,
			theta_x_in_lab.c_str());
	lost_part_tree->Branch(br_y_in_lab.c_str(), &y_in, y_in_lab.c_str());
	lost_part_tree->Branch(br_theta_y_in_lab.c_str(), &theta_y_in,
			theta_y_in_lab.c_str());
	lost_part_tree->Branch(br_ksi_in_lab.c_str(), &ksi_in, ksi_in_lab.c_str());
	lost_part_tree->Branch(br_s_in_lab.c_str(), &s_in, s_in_lab.c_str());

	lost_part_tree->Branch(br_x_out_lab.c_str(), &x_out, x_out_lab.c_str());
	lost_part_tree->Branch(br_theta_x_out_lab.c_str(), &theta_x_out,
			theta_x_out_lab.c_str());
	lost_part_tree->Branch(br_y_out_lab.c_str(), &y_out, y_out_lab.c_str());
	lost_part_tree->Branch(br_theta_y_out_lab.c_str(), &theta_y_out,
			theta_y_out_lab.c_str());
	lost_part_tree->Branch(br_ksi_out_lab.c_str(), &ksi_out,
			ksi_out_lab.c_str());
	lost_part_tree->Branch(br_s_out_lab.c_str(), &s_out, s_out_lab.c_str());
	lost_part_tree->Branch(br_element_out_lab.c_str(), element_out,
			element_out_lab.c_str());

	lost_part_tree->Write(NULL, TObject::kOverwrite);
	return lost_part_tree;
}

void MADParamGenerator::ClearWorkingFiles(
		const Parametisation_configuration &conf,
		std::string sample_file_name) {
	std::string cmd;
	cmd = "rm -f ./part.in ./trackone ./trackloss ./" + sample_file_name;
	system(cmd.c_str());
}

void MADParamGenerator::DeleteLostParticlesFiles(
		const Parametisation_configuration &conf) {
	std::string cmd;
	cmd = "rm -f ./" + conf.lost_particles_tree_filename + " ./"
			+ conf.lost_particles_hist_filename;
	system(cmd.c_str());
}

void MADParamGenerator::DeleteApertureTestFiles(
		const Parametisation_configuration &conf) {
	std::string cmd;
	cmd = "rm -f ./" + conf.samples_aperture_test_file_name;
	system(cmd.c_str());
}


std::vector<std::string> MADParamGenerator::GetAllScoringPlanes(int id) {
	std::vector < std::string > scoring_planes;
	Parametisation_configuration conf = GetParamConfiguration(id);

	scoring_planes.push_back(conf.to_marker_name);

	for (int i = 0; i < conf.inter_planes.size(); i++)
		scoring_planes.push_back(conf.inter_planes[i].to_marker_name);

	return scoring_planes;
}

int MADParamGenerator::RunMADXWithExternalData(int config_id, int size) {
	Parametisation_configuration conf = GetParamConfiguration(config_id);

	std::string cmd;
	cmd = "rm -f ./trackone ./trackloss";
	system(cmd.c_str());

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);
	std::string confFile = conf.base_mad_conf_file;
	if (conf.aperture_limit) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}
	mad_conf_gen.GenerateMADConfFile(confFile,
			conf.processed_mad_conf_file, conf.from_marker_name,
			conf.from_marker_s_pos, conf.define_from, conf.to_marker_name,
			conf.to_marker_s_pos, conf.define_to, size, conf.aperture_limit,
			aperture_markers, conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Running MADX from " << conf.from_marker_name
			<< " to " << conf.to_marker_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
	//exit(0);
	return size;
}

int MADParamGenerator::BuildSample(const Parametisation_configuration &conf,
		std::string sample_file_name, bool recloss, bool compare_apert,
		int sample_type) {
	ClearWorkingFiles(conf, sample_file_name);
	if (recloss && !compare_apert)
		DeleteLostParticlesFiles(conf);

	if (compare_apert)
		DeleteApertureTestFiles(conf);

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);
	std::string confFile = conf.base_mad_conf_file; //if aperture or recloss we need to use mad thin_track module
	if (conf.aperture_limit || recloss || compare_apert) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}
	mad_conf_gen.GenerateMADConfFile(confFile, conf.processed_mad_conf_file,
			conf.from_marker_name, conf.from_marker_s_pos, conf.define_from,
			conf.to_marker_name, conf.to_marker_s_pos, conf.define_to,
			conf.number_of_part_per_sample,
			(conf.aperture_limit || recloss || compare_apert), aperture_markers,
			conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Generating random samples, from "
			<< conf.from_marker_name << " to " << conf.to_marker_name
			<< ", file " << sample_file_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	do {
		if (sample_type == 0) {
			std::cout << "Flat sample generated." << std::endl;
			mad_conf_gen.GenerateRandomSamples(conf.number_of_part_per_sample,
					conf.x_min, conf.x_max, conf.theta_x_min, conf.theta_x_max,
					conf.y_min, conf.y_max, conf.theta_y_min, conf.theta_y_max,
					conf.ksi_min, conf.ksi_max, "part.in");
		} else if (sample_type == 1) {
			std::cout << "1/xi, 1/t sample generated." << std::endl;
			GenerateDiffractiveProtonsSamples(conf.number_of_part_per_sample,
					conf.nominal_beam_energy, conf.x_min, conf.x_max,
					conf.theta_x_min, conf.theta_x_max, conf.y_min, conf.y_max,
					conf.theta_y_min, conf.theta_y_max, conf.ksi_min,
					conf.ksi_max, "part.in");
		} else if (sample_type == 2) {
			std::cout << "1/t, xi=0 sample generated." << std::endl;
			GenerateElasticProtonsSamples(conf.number_of_part_per_sample,
					conf.nominal_beam_energy, conf.x_min, conf.x_max,
					conf.theta_x_min, conf.theta_x_max, conf.y_min, conf.y_max,
					conf.theta_y_min, conf.theta_y_max, conf.ksi_min,
					conf.ksi_max, "part.in");
		}

		mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
		current_iteration_particles = mad_conf_gen.AppendRootTree(
				sample_file_name, conf.destination_branch_prefix,
				conf.to_marker_name, recloss && !compare_apert,
				conf.lost_particles_tree_filename, aperture_markers,
				compare_apert);

		if (!recloss || compare_apert) {
			total_generated_particles = GetNumberOfEntries(sample_file_name,
					conf.destination_branch_prefix);
			std::cout << "Total number of particles arrived at "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		} else {
			total_generated_particles = GetLostParticlesEntries(conf);
			std::cout << "Total number of particles lost before "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		}
	} while (total_generated_particles < conf.tot_entries_number
			&& current_iteration_particles > 5);

	if (!recloss) {
		PrintTreeInfo(conf, sample_file_name);
	}

	return total_generated_particles;
}

int MADParamGenerator::BuildDebugSample(
		const Parametisation_configuration &conf, std::string sample_file_name,
		bool recloss, bool compare_apert) {
	ClearWorkingFiles(conf, sample_file_name);
	if (recloss && !compare_apert)
		DeleteLostParticlesFiles(conf);

	if (compare_apert)
		DeleteApertureTestFiles(conf);

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);

	std::string confFile = conf.base_mad_conf_file; //if aperture or recloss we need to use mad thin_track module
	if (conf.aperture_limit || recloss || compare_apert) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}

	mad_conf_gen.GenerateMADConfFile(confFile, conf.processed_mad_conf_file,
			conf.from_marker_name, conf.from_marker_s_pos, conf.define_from,
			conf.to_marker_name, conf.to_marker_s_pos, conf.define_to,
			conf.number_of_part_per_sample,
			(conf.aperture_limit || recloss || compare_apert), aperture_markers,
			conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Generating random samples, from "
			<< conf.from_marker_name << " to " << conf.to_marker_name
			<< ", file " << sample_file_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	do {
		mad_conf_gen.GenerateDebugRandomSamples(conf.number_of_part_per_sample,
				conf.x_min, conf.x_max, conf.theta_x_min, conf.theta_x_max,
				conf.y_min, conf.y_max, conf.theta_y_min, conf.theta_y_max,
				conf.ksi_min, conf.ksi_max, "part.in");
		mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
		current_iteration_particles = mad_conf_gen.AppendRootTree(
				sample_file_name, conf.destination_branch_prefix,
				conf.to_marker_name, recloss && !compare_apert,
				conf.lost_particles_tree_filename, aperture_markers,
				compare_apert);

		if (!recloss || compare_apert) {
			total_generated_particles = GetNumberOfEntries(sample_file_name,
					conf.destination_branch_prefix);
			std::cout << "Total number of particles arrived at "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		} else {
			total_generated_particles = GetLostParticlesEntries(conf);
			std::cout << "Total number of particles lost before "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		}
	} while (total_generated_particles < conf.tot_entries_number
			&& current_iteration_particles > 5);

	if (!recloss) {
		PrintTreeInfo(conf, sample_file_name);
	}

	return total_generated_particles;
}

int MADParamGenerator::BuildGridDebugSample(
		const Parametisation_configuration &conf, std::string sample_file_name,
		bool recloss, bool compare_apert) {
	ClearWorkingFiles(conf, sample_file_name);
	if (recloss && !compare_apert)
		DeleteLostParticlesFiles(conf);

	if (compare_apert)
		DeleteApertureTestFiles(conf);

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);
	std::string confFile = conf.base_mad_conf_file; //if aperture or recloss we need to use mad thin_track module
	if (conf.aperture_limit || recloss || compare_apert) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}
	mad_conf_gen.GenerateMADConfFile(confFile, conf.processed_mad_conf_file,
			conf.from_marker_name, conf.from_marker_s_pos, conf.define_from,
			conf.to_marker_name, conf.to_marker_s_pos, conf.define_to,
			conf.number_of_part_per_sample,
			(conf.aperture_limit || recloss || compare_apert), aperture_markers,
			conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Generating random samples, from "
			<< conf.from_marker_name << " to " << conf.to_marker_name
			<< ", file " << sample_file_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	do {
		mad_conf_gen.GenerateGridDebugRandomSamples(
				conf.number_of_part_per_sample, conf.x_min, conf.x_max,
				conf.theta_x_min, conf.theta_x_max, conf.y_min, conf.y_max,
				conf.theta_y_min, conf.theta_y_max, conf.ksi_min, conf.ksi_max,
				"part.in");
		mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
		current_iteration_particles = mad_conf_gen.AppendRootTree(
				sample_file_name, conf.destination_branch_prefix,
				conf.to_marker_name, recloss && !compare_apert,
				conf.lost_particles_tree_filename, aperture_markers,
				compare_apert);

		if (!recloss || compare_apert) {
			total_generated_particles = GetNumberOfEntries(sample_file_name,
					conf.destination_branch_prefix);
			std::cout << "Total number of particles arrived at "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		} else {
			total_generated_particles = GetLostParticlesEntries(conf);
			std::cout << "Total number of particles lost before "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		}
	} while (total_generated_particles < conf.tot_entries_number
			&& current_iteration_particles > 5);

	if (!recloss) {
		PrintTreeInfo(conf, sample_file_name);
	}

	return total_generated_particles;
}

int MADParamGenerator::BuildXiContTDiscPhiContDebugSample(
		const Parametisation_configuration &conf, std::string sample_file_name,
		bool recloss, bool compare_apert) {
	ClearWorkingFiles(conf, sample_file_name);
	if (recloss && !compare_apert)
		DeleteLostParticlesFiles(conf);

	if (compare_apert)
		DeleteApertureTestFiles(conf);

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);
	std::string confFile = conf.base_mad_conf_file; //if aperture or recloss we need to use mad thin_track module
	if (conf.aperture_limit || recloss || compare_apert) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}

	mad_conf_gen.GenerateMADConfFile(confFile, conf.processed_mad_conf_file,
			conf.from_marker_name, conf.from_marker_s_pos, conf.define_from,
			conf.to_marker_name, conf.to_marker_s_pos, conf.define_to,
			conf.number_of_part_per_sample,
			(conf.aperture_limit || recloss || compare_apert), aperture_markers,
			conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Generating random samples, from "
			<< conf.from_marker_name << " to " << conf.to_marker_name
			<< ", file " << sample_file_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	do {
		mad_conf_gen.GenerateXiContTDiscPhiContDebugRandomSamples(
				conf.number_of_part_per_sample, conf.x_min, conf.x_max,
				conf.theta_x_min, conf.theta_x_max, conf.y_min, conf.y_max,
				conf.theta_y_min, conf.theta_y_max, conf.ksi_min, conf.ksi_max,
				"part.in");
		mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
		current_iteration_particles = mad_conf_gen.AppendRootTree(
				sample_file_name, conf.destination_branch_prefix,
				conf.to_marker_name, recloss && !compare_apert,
				conf.lost_particles_tree_filename, aperture_markers,
				compare_apert);

		if (!recloss || compare_apert) {
			total_generated_particles = GetNumberOfEntries(sample_file_name,
					conf.destination_branch_prefix);
			std::cout << "Total number of particles arrived at "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		} else {
			total_generated_particles = GetLostParticlesEntries(conf);
			std::cout << "Total number of particles lost before "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		}
	} while (total_generated_particles < conf.tot_entries_number
			&& current_iteration_particles > 5);

	if (!recloss) {
		PrintTreeInfo(conf, sample_file_name);
	}

	return total_generated_particles;
}

int MADParamGenerator::BuildDiffractiveProtonsSample(
		const Parametisation_configuration &conf, std::string sample_file_name,
		bool recloss, bool compare_apert) {
	ClearWorkingFiles(conf, sample_file_name);
	if (recloss && !compare_apert)
		DeleteLostParticlesFiles(conf);

	if (compare_apert)
		DeleteApertureTestFiles(conf);

	MADParamGenerator mad_conf_gen;

	std::vector < std::string > aperture_markers;
	for (int i = 0; i < conf.inter_planes.size(); i++)
		aperture_markers.push_back(conf.inter_planes[i].to_marker_name);
	std::string confFile = conf.base_mad_conf_file; //if aperture or recloss we need to use mad thin_track module
	if (conf.aperture_limit || recloss || compare_apert) { //instead of ptc thick_track
		confFile = conf.base_mad_thin_conf_file;
	}

	mad_conf_gen.GenerateMADConfFile(confFile, conf.processed_mad_conf_file,
			conf.from_marker_name, conf.from_marker_s_pos, conf.define_from,
			conf.to_marker_name, conf.to_marker_s_pos, conf.define_to,
			conf.number_of_part_per_sample,
			(conf.aperture_limit || recloss || compare_apert), aperture_markers,
			conf.beam);
	int total_generated_particles = 0;
	int current_iteration_particles = 0;

	std::cout << std::endl << "Generating random samples, from "
			<< conf.from_marker_name << " to " << conf.to_marker_name
			<< ", file " << sample_file_name << std::endl;
	std::cout << "Number of inter-planes: " << conf.inter_planes.size()
			<< std::endl;

	do {
		mad_conf_gen.GenerateDiffractiveProtonsSamples(
				conf.number_of_part_per_sample, conf.nominal_beam_energy,
				conf.x_min, conf.x_max, conf.theta_x_min, conf.theta_x_max,
				conf.y_min, conf.y_max, conf.theta_y_min, conf.theta_y_max,
				conf.ksi_min, conf.ksi_max, "part.in");
		mad_conf_gen.RunMAD(conf.processed_mad_conf_file);
		current_iteration_particles = mad_conf_gen.AppendRootTree(
				sample_file_name, conf.destination_branch_prefix,
				conf.to_marker_name, recloss && !compare_apert,
				conf.lost_particles_tree_filename, aperture_markers,
				compare_apert);

		if (!recloss || compare_apert) {
			total_generated_particles = GetNumberOfEntries(sample_file_name,
					conf.destination_branch_prefix);
			std::cout << "Total number of particles arrived at "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		} else {
			total_generated_particles = GetLostParticlesEntries(conf);
			std::cout << "Total number of particles lost before "
					<< conf.to_marker_name << " " << total_generated_particles
					<< " of " << conf.tot_entries_number << std::endl;
		}
	} while (total_generated_particles < conf.tot_entries_number
			&& current_iteration_particles > 5);

	if (!recloss) {
		PrintTreeInfo(conf, sample_file_name);
	}

	return total_generated_particles;
}

void MADParamGenerator::PrintTreeInfo(const Parametisation_configuration &conf,
		std::string sample_file_name) {
	TFile *f = TFile::Open(sample_file_name.c_str(), "read");
	if (!f->IsOpen())
		return;

	TTree *tree = (TTree*) f->Get("transport_samples");
	if (!tree)
		return;

	tree->Print();
	f->Close();
	delete f;
}

TTree *MADParamGenerator::GetSamplesTree(
		const Parametisation_configuration &conf, std::string sample_file_name,
		TFile *&f) {
	f = TFile::Open(sample_file_name.c_str(), "read");
	if (!f->IsOpen())
		return NULL;

	TTree *tree = (TTree*) f->Get("transport_samples");
	if (!tree)
		return NULL;
	return tree;
}

TTree *MADParamGenerator::GetAccelAcceptTree(TFile *f) {
	if (!f || !f->IsOpen())
		return NULL;

	TTree *tree = (TTree*) f->Get("acc_acept_tree");
	if (!tree)
		return NULL;

	return tree;
}

void MADParamGenerator::WriteAccelAcceptTree(TFile *f, TTree *acc_acept_tree) {
	if (!f->IsOpen())
		return;

	f->cd();
	if (!acc_acept_tree)
		return;

	acc_acept_tree->Write(NULL, TObject::kOverwrite);
	acc_acept_tree->Print();
}

int MADParamGenerator::GenerateTrainingData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "==== Training data being generated ====" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << std::endl;
	return BuildSample(conf, conf.samples_train_root_file_name);
}

int MADParamGenerator::GenerateDebugData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "==== Debug data being generated ====" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << std::endl;
	return BuildDebugSample(conf, conf.samples_train_root_file_name);
}

int MADParamGenerator::GenerateGridDebugData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "==== Debug data being generated ====" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << std::endl;
	return BuildGridDebugSample(conf, conf.samples_train_root_file_name);
}

int MADParamGenerator::GenerateDiffractiveProtonsData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "==== Debug data being generated ====" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << std::endl;
	return BuildDiffractiveProtonsSample(conf,
			conf.samples_train_root_file_name);
}

int MADParamGenerator::GenerateXiContTDiscPhiContDebugData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << "==== Debug data being generated ====" << std::endl;
	std::cout << "=======================================" << std::endl;
	std::cout << std::endl;
	return BuildXiContTDiscPhiContDebugSample(conf,
			conf.samples_train_root_file_name);
}

int MADParamGenerator::GenerateTestingData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "======================================" << std::endl;
	std::cout << "==== Testing data being generated ====" << std::endl;
	std::cout << "======================================" << std::endl;
	std::cout << std::endl;
	return BuildSample(conf, conf.samples_test_root_file_name);
}

int MADParamGenerator::GenerateLostParticleData(
		const Parametisation_configuration &conf, int type) {
	std::cout << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "==== Lost particle data being generated ====" << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << std::endl;
	return BuildSample(conf, conf.samples_test_root_file_name, true, false,
			type);
}

int MADParamGenerator::GenerateApertureTestingData(
		const Parametisation_configuration &conf) {
	std::cout << std::endl;
	std::cout << "=================================================="
			<< std::endl;
	std::cout << "==== Aperture model test data being generated ===="
			<< std::endl;
	std::cout << "=================================================="
			<< std::endl;
	std::cout << std::endl;
	return BuildSample(conf, conf.samples_aperture_test_file_name, false, true);
}

void MADParamGenerator::OpenXMLConfigurationFile(std::string file_name) {
	xml_parser.read(file_name);
	xml_parser.setFilename(file_name);
	xml_parser.read();
}

Parametisation_configuration MADParamGenerator::GetParamConfiguration(int id) {
	Parametisation_configuration conf;
	conf.base_mad_conf_file = xml_parser.get < std::string
			> (id, std::string("base_mad_conf_file"));
	conf.processed_mad_conf_file = xml_parser.get < std::string
			> (id, "processed_mad_conf_file");
	conf.base_mad_thin_conf_file = xml_parser.get < std::string
			> (id, "base_mad_thin_conf_file");
	conf.beam = xml_parser.get < std::string > (id, "beam");
	conf.nominal_beam_energy = xml_parser.get<double>(id,
			"nominal_beam_energy");

	conf.from_marker_name = xml_parser.get < std::string
			> (id, "from_marker_name");
	std::transform(conf.from_marker_name.begin(), conf.from_marker_name.end(),
			conf.from_marker_name.begin(), ::tolower);

	conf.from_marker_s_pos = xml_parser.get<double>(id, "from_marker_s_pos");
	conf.define_from = xml_parser.get<int>(id, "define_from");

	conf.to_marker_name = xml_parser.get < std::string > (id, "to_marker_name");
	std::transform(conf.to_marker_name.begin(), conf.to_marker_name.end(),
			conf.to_marker_name.begin(), ::tolower);

	conf.to_marker_s_pos = xml_parser.get<double>(id, "to_marker_s_pos");
	conf.define_to = xml_parser.get<int>(id, "define_to");
	conf.aperture_limit = xml_parser.get<int>(id, "aperture_limit");
	conf.tot_entries_number = xml_parser.get<int>(id, "tot_entries_number");

	conf.number_of_part_per_sample = xml_parser.get<int>(id,
			"number_of_part_per_sample");
	conf.x_min = xml_parser.get<double>(id, "x_min");
	conf.x_max = xml_parser.get<double>(id, "x_max");
	conf.theta_x_min = xml_parser.get<double>(id, "theta_x_min");
	conf.theta_x_max = xml_parser.get<double>(id, "theta_x_max");
	conf.y_min = xml_parser.get<double>(id, "y_min");
	conf.y_max = xml_parser.get<double>(id, "y_max");
	conf.theta_y_min = xml_parser.get<double>(id, "theta_y_min");
	conf.theta_y_max = xml_parser.get<double>(id, "theta_y_max");
	conf.ksi_min = xml_parser.get<double>(id, "ksi_min");
	conf.ksi_max = xml_parser.get<double>(id, "ksi_max");

	conf.samples_train_root_file_name = xml_parser.get < std::string
			> (id, "samples_train_root_file_name");
	conf.samples_test_root_file_name = xml_parser.get < std::string
			> (id, "samples_test_root_file_name");
	conf.samples_aperture_test_file_name = xml_parser.get < std::string
			> (id, "samples_aperture_test_file_name");

	conf.destination_branch_prefix = xml_parser.get < std::string
			> (id, "destination_branch_prefix");
	std::transform(conf.destination_branch_prefix.begin(),
			conf.destination_branch_prefix.end(),
			conf.destination_branch_prefix.begin(), ::tolower);

	std::string pol_type = xml_parser.get < std::string
			> (id, "polynomials_type");
	if (pol_type == "kMonomials")
		conf.polynomials_type = TMultiDimFet::kMonomials;
	else if (pol_type == "kChebyshev")
		conf.polynomials_type = TMultiDimFet::kChebyshev;
	else if (pol_type == "kLegendre")
		conf.polynomials_type = TMultiDimFet::kLegendre;
	else
		conf.polynomials_type = TMultiDimFet::kMonomials;

	std::string sel_mode = xml_parser.get < std::string
			> (id, "terms_selelection_mode");
	if (sel_mode == "AUTOMATIC")
		conf.terms_selelection_mode = LHCOpticsApproximator::AUTOMATIC;
	else if (sel_mode == "PREDEFINED")
		conf.terms_selelection_mode = LHCOpticsApproximator::PREDEFINED;
	else
		conf.terms_selelection_mode = LHCOpticsApproximator::AUTOMATIC;

	conf.max_degree_x = xml_parser.get<int>(id, "max_degree_x");
	conf.max_degree_tx = xml_parser.get<int>(id, "max_degree_tx");
	conf.max_degree_y = xml_parser.get<int>(id, "max_degree_y");
	conf.max_degree_ty = xml_parser.get<int>(id, "max_degree_ty");
	conf.common_terms = xml_parser.get<int>(id, "common_terms");

	conf.precision_x = xml_parser.get<double>(id, "precision_x");
	conf.precision_tx = xml_parser.get<double>(id, "precision_tx");
	conf.precision_y = xml_parser.get<double>(id, "precision_y");
	conf.precision_ty = xml_parser.get<double>(id, "precision_ty");

	conf.approximation_error_histogram_file = xml_parser.get < std::string
			> (id, "approximation_error_histogram_file");

	conf.lost_particles_tree_filename = xml_parser.get < std::string
			> (id, "lost_particles_tree_filename");
	conf.lost_particles_hist_filename = xml_parser.get < std::string
			> (id, "lost_particles_hist_filename");

	conf.optics_parametrisation_file = xml_parser.get < std::string
			> (id, "optics_parametrisation_file");
	conf.optics_parametrisation_name = xml_parser.get < std::string
			> (id, "optics_parametrisation_name");

	std::vector<int> ids = xml_parser.getIds(id);
	for (int i = 0; i < ids.size(); i++)
		conf.inter_planes.push_back(GetApertureConfiguration(id, ids[i]));

	return conf;
}

Parametisation_aperture_configuration MADParamGenerator::GetApertureConfiguration(
		int param_id, int apreture_id) {
	Parametisation_aperture_configuration conf;

	conf.to_marker_name = xml_parser.get < std::string
			> (param_id, apreture_id, std::string("to_marker_name"));
	std::transform(conf.to_marker_name.begin(), conf.to_marker_name.end(),
			conf.to_marker_name.begin(), ::tolower);

	std::string ap = xml_parser.get < std::string
			> (param_id, apreture_id, "ap_type");
	if (ap == "RECTELLIPSE")
		conf.ap_type = LHCApertureApproximator::RECTELLIPSE;
	else
		conf.ap_type = LHCApertureApproximator::RECTELLIPSE;

	conf.rect_rx = xml_parser.get<double>(param_id, apreture_id, "rect_rx");
	conf.rect_ry = xml_parser.get<double>(param_id, apreture_id, "rect_ry");
	conf.el_rx = xml_parser.get<double>(param_id, apreture_id, "el_rx");
	conf.el_ry = xml_parser.get<double>(param_id, apreture_id, "el_ry");

	return conf;
}

bool MADParamGenerator::CheckParamConfId(int id) {
	std::vector<int> ids = xml_parser.getIds();
	bool val = false;
	for (int i = 0; !val && i < ids.size(); i++)
		if (ids[i] == id)
			val = true;
	return val;
}

bool MADParamGenerator::CheckApertureConfId(int param_id, int apreture_id) {
	std::vector<int> ids = xml_parser.getIds(param_id);
	bool val = false;
	for (int i = 0; !val && i < ids.size(); i++)
		if (ids[i] == apreture_id)
			val = true;
	return val;
}

void MADParamGenerator::MakeParametrization(int id, bool generate_samples) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);

	if (generate_samples) {
		this->GenerateTrainingData(conf);
		this->GenerateTestingData(conf);
		this->GenerateApertureTestingData(conf);
	}
	//  exit(0);

	std::string name = conf.optics_parametrisation_name;
	LHCOpticsApproximator approximator(name, name, conf.polynomials_type,
			conf.beam, conf.nominal_beam_energy);

	double prec[4];
	prec[0] = conf.precision_x;
	prec[1] = conf.precision_tx;
	prec[2] = conf.precision_y;
	prec[3] = conf.precision_ty;

	PrintCurrentMemoryUsage(
			"MADParamGenerator::MakeParametrization, before approximator.Train");
	TFile *train_file;
	approximator.Train(
			this->GetSamplesTree(conf, conf.samples_train_root_file_name,
					train_file), conf.destination_branch_prefix,
			conf.terms_selelection_mode, conf.max_degree_x, conf.max_degree_tx,
			conf.max_degree_y, conf.max_degree_ty, conf.common_terms, prec);
	train_file->Close();
	delete train_file;

	PrintCurrentMemoryUsage(
			"MADParamGenerator::MakeParametrization, before new TFile");
	TFile *f = TFile::Open(conf.approximation_error_histogram_file.c_str(),
			"update");
	PrintCurrentMemoryUsage(
			"MADParamGenerator::MakeParametrization, before approximator.Test");
	TFile *test_file;
	approximator.Test(
			this->GetSamplesTree(conf, conf.samples_test_root_file_name,
					test_file), f, conf.destination_branch_prefix, "");
	test_file->Close();
	delete test_file;
	f->Close();
	delete f;

	PrintCurrentMemoryUsage(
			"MADParamGenerator::MakeParametrization, before TrainAndAddApertures");
	TrainAndAddApertures(conf, approximator,
			conf.approximation_error_histogram_file.c_str());
	PrintCurrentMemoryUsage(
			"MADParamGenerator::MakeParametrization, after TrainAndAddApertures");

	if (conf.samples_aperture_test_file_name != "") {
		TFile *acc_accept_file = TFile::Open(
				conf.samples_aperture_test_file_name.c_str(), "update");
		TTree *new_acc_accept_tree = CreateAccelAcceptTree(acc_accept_file,
				"mad_param_filled_together");
		approximator.TestAperture(GetAccelAcceptTree(acc_accept_file),
				new_acc_accept_tree);
		WriteAccelAcceptTree(acc_accept_file, new_acc_accept_tree);
		acc_accept_file->Close();
		delete acc_accept_file;
	}

	TFile *approx_out = TFile::Open(conf.optics_parametrisation_file.c_str(),
			"update");
	approximator.Write(NULL, TObject::kOverwrite);
	approx_out->Close();
	delete approx_out;
}

void MADParamGenerator::GenerateDebugSamples(int id) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);

	this->GenerateDebugData(conf);
}

void MADParamGenerator::GenerateGridDebugSamples(int id) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);

	this->GenerateGridDebugData(conf);
}

void MADParamGenerator::GenerateXiContTDiscPhiContDebugSamples(int id) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);

	this->GenerateXiContTDiscPhiContDebugData(conf);
}

void MADParamGenerator::GenerateDiffractiveProtons(int id) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);

	this->GenerateDiffractiveProtonsData(conf);
}

void MADParamGenerator::TrainAndAddApertures(
		const Parametisation_configuration &conf,
		LHCOpticsApproximator &approximator, const char * f_out_name) {
	PrintCurrentMemoryUsage(
			"MADParamGenerator::TrainAndAddApertures, begining");

	for (int i = 0; i < conf.inter_planes.size(); i++) {
		PrintCurrentMemoryUsage(
				"MADParamGenerator::TrainAndAddApertures, for, begining");
		std::string name = conf.from_marker_name + "_to_"
				+ conf.inter_planes[i].to_marker_name;

		pid_t cpid, w;
		int status;
		cpid = fork();
		if (cpid == -1) {
			perror("fork");
			exit(EXIT_FAILURE);
		}
		if (cpid == 0) {
			LHCOpticsApproximator aper_approx(name, name, conf.polynomials_type,
					conf.beam, conf.nominal_beam_energy);

			PrintCurrentMemoryUsage(
					"MADParamGenerator::TrainAndAddApertures, for, new approximator created");

			double prec[4];
			prec[0] = conf.precision_x;
			prec[1] = conf.precision_tx;
			prec[2] = conf.precision_y;
			prec[3] = conf.precision_ty;

			PrintCurrentMemoryUsage(
					"MADParamGenerator::TrainAndAddApertures, for, before aper_approx.Train");
			TFile *train_file;
			aper_approx.Train(
					this->GetSamplesTree(conf,
							conf.samples_train_root_file_name, train_file),
					conf.inter_planes[i].to_marker_name,
					conf.terms_selelection_mode, conf.max_degree_x,
					conf.max_degree_tx, conf.max_degree_y, conf.max_degree_ty,
					conf.common_terms, prec);
			train_file->Close();
			delete train_file;

			PrintCurrentMemoryUsage(
					"MADParamGenerator::TrainAndAddApertures, for, before aper_approx.Test");
			TFile *f_out = TFile::Open(f_out_name, "update");
			TFile *test_file;
			aper_approx.Test(
					this->GetSamplesTree(conf, conf.samples_test_root_file_name,
							test_file), f_out,
					conf.inter_planes[i].to_marker_name,
					conf.optics_parametrisation_name);
			test_file->Close();
			delete test_file;
			f_out->Close();
			delete f_out;

			std::cout << "Aperture being written to a temp file: "
					<< aper_approx.GetName() << std::endl;
			TFile *temp_outputfile = TFile::Open(
					"___temp_aper_file_to_be_deleted__.root", "RECREATE");
			temp_outputfile->cd();
			aper_approx.Write("temporary_aperture_object", TObject::kOverwrite);
			temp_outputfile->Close();
			delete temp_outputfile;
			exit(EXIT_SUCCESS);
		}

		do {
			w = waitpid(cpid, &status, WUNTRACED | WCONTINUED);
			if (w == -1) {
				perror("waitpid");
				exit(EXIT_FAILURE);
			}
			if (WIFEXITED(status)) {
				printf("exited, status=%d\n", WEXITSTATUS(status));
			} else if (WIFSIGNALED(status)) {
				printf("killed by signal %d\n", WTERMSIG(status));
			} else if (WIFSTOPPED(status)) {
				printf("stopped by signal %d\n", WSTOPSIG(status));
			} else if (WIFCONTINUED(status)) {
				printf("continued\n");
			}
		} while (!WIFEXITED(status) && !WIFSIGNALED(status));

		TFile *temp_inputfile = TFile::Open(
				"___temp_aper_file_to_be_deleted__.root", "READ");
		temp_inputfile->cd();
		LHCOpticsApproximator aper_approx =
				*((LHCOpticsApproximator*) temp_inputfile->Get(
						"temporary_aperture_object"));
		std::cout << "Read_aper_name: " << aper_approx.GetName() << std::endl;

		PrintCurrentMemoryUsage(
				"MADParamGenerator::TrainAndAddApertures, for, approximator.AddRectEllipseAperture");
		approximator.AddRectEllipseAperture(aper_approx,
				conf.inter_planes[i].rect_rx, conf.inter_planes[i].rect_ry,
				conf.inter_planes[i].el_rx, conf.inter_planes[i].el_ry);
		temp_inputfile->Close();
		delete temp_inputfile;
	}
	PrintCurrentMemoryUsage("MADParamGenerator::TrainAndAddApertures, end");
}

void MADParamGenerator::MakeAllParametrizations(bool generate_samples) {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		MakeParametrization(ids[i], generate_samples);
	}
}

void MADParamGenerator::GenerateAllDebugSamples() {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		GenerateDebugSamples(ids[i]);
	}
}

void MADParamGenerator::GenerateAllGridDebugSamples() {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		GenerateGridDebugSamples(ids[i]);
	}
}

void MADParamGenerator::GenerateAllXiContTDiscPhiContDebugSamples() {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		GenerateXiContTDiscPhiContDebugSamples(ids[i]);
	}
}

void MADParamGenerator::GenerateAllDiffractiveProtons() {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		GenerateDiffractiveProtons(ids[i]);
	}
}

void MADParamGenerator::IdentifyApertures(int id, int dist_type) {
	if (!CheckParamConfId(id))
		return;

	Parametisation_configuration conf = this->GetParamConfiguration(id);
	this->GenerateLostParticleData(conf, dist_type);
}

void MADParamGenerator::IdentifyAperturesForAll(int dist_type) {
	std::vector<int> ids = xml_parser.getIds();

	for (int i = 0; i < ids.size(); i++) {
		IdentifyApertures(ids[i], dist_type);
	}
}

std::ostream & operator<<(std::ostream &s, const Parametisation_configuration &c) {
	s << "base_mad_conf_file " << c.base_mad_conf_file << std::endl;
	s << "processed_mad_conf_file " << c.processed_mad_conf_file << std::endl;
	s << "from_marker_name " << c.from_marker_name << std::endl;
	s << "from_marker_s_pos " << c.from_marker_s_pos << std::endl;
	s << "define_from " << c.define_from << std::endl;
	s << "to_marker_name " << c.to_marker_name << std::endl;
	s << "to_marker_s_pos " << c.to_marker_s_pos << std::endl;
	s << "define_to " << c.define_to << std::endl;
	s << "aperture_limit " << c.aperture_limit << std::endl;
	s << "tot_entries_number " << c.tot_entries_number << std::endl;

	s << "number_of_part_per_sample " << c.number_of_part_per_sample
			<< std::endl;
	s << "x_min " << c.x_min << std::endl;
	s << "x_max " << c.x_max << std::endl;
	s << "theta_x_min " << c.theta_x_min << std::endl;
	s << "theta_x_max " << c.theta_x_max << std::endl;
	s << "y_min " << c.y_min << std::endl;
	s << "y_max " << c.y_max << std::endl;
	s << "theta_y_min " << c.theta_y_min << std::endl;
	s << "theta_y_max " << c.theta_y_max << std::endl;
	s << "ksi_min " << c.ksi_min << std::endl;
	s << "ksi_max " << c.ksi_max << std::endl;

	s << "samples_train_root_file_name " << c.samples_train_root_file_name
			<< std::endl;
	s << "samples_test_root_file_name " << c.samples_test_root_file_name
			<< std::endl;
	s << "destination_branch_prefix " << c.destination_branch_prefix
			<< std::endl;

	s << "polynomials_type " << c.polynomials_type << std::endl;
	s << "terms_selelection_mode " << c.terms_selelection_mode << std::endl;
	s << "max_degree_x " << c.max_degree_x << std::endl;
	s << "max_degree_tx " << c.max_degree_tx << std::endl;
	s << "max_degree_y " << c.max_degree_y << std::endl;
	s << "max_degree_ty " << c.max_degree_ty << std::endl;
	s << "common_terms " << c.common_terms << std::endl;
	s << "approximation_error_histogram_file "
			<< c.approximation_error_histogram_file << std::endl;

	s << "lost_particles_tree_filename " << c.lost_particles_tree_filename
			<< std::endl;
	s << "lost_particles_hist_filename " << c.lost_particles_hist_filename
			<< std::endl;

	s << "optics_parametrisation_file " << c.optics_parametrisation_file
			<< std::endl;
	s << "inter_planes " << c.inter_planes.size() << std::endl;
}

std::ostream & operator<<(std::ostream &s,
		const Parametisation_aperture_configuration &c) {
	s << "to_marker_name " << c.to_marker_name << std::endl;
	s << "ap_type " << c.ap_type << std::endl;
	s << "rect_rx " << c.rect_rx << std::endl;
	s << "rect_ry " << c.rect_ry << std::endl;
	s << "el_rx " << c.el_rx << std::endl;
	s << "el_ry " << c.el_ry << std::endl;
}
