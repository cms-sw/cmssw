#ifndef PdgIdConverter_h
#define PdgIdConverter_h

#include <string>
#include <sstream>
#include <iostream>
using namespace std;

//#include "ConvertToString.h"

template <class T>
std::string to_string (const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

string ParticleName(int iPdgId);

#endif // #ifndef PdgIdConverter_h

#ifdef PdgIdConverter_cxx

string ParticleName(int iPdgId) {
	
	string name = to_string( iPdgId );
	
	switch( iPdgId ) {
		
		// Quark codes
		
		case 1:
		name = "d";
		break;
		
		case -1:
		name = "dbar";
		break;
		
		case 2:
		name = "u";
		break;
		
		case -2:
		name = "ubar";
		break;
		
		case 3:
		name = "s";
		break;
		
		case -3:
		name = "sbar";
		break;
		
		case 4:
		name = "c";
		break;
		
		case -4:
		name = "cbar";
		break;
		
		case 5:
		name = "b";
		break;
		
		case -5:
		name = "bbar";
		break;
		
		case 6:
		name = "t";
		break;
		
		case -6:
		name = "tbar";
		break;
		
		case 7:
		name = "b'";
		break;
		
		case -7:
		name = "bbar'";
		break;
		
		case 8:
		name = "t'";
		break;
		
		case -8:
		name = "tbar'";
		break;
		
		// Lepton codes
		
		case 11:
		name = "e-";
		break;
		
		case -11:
		name = "e+";
		break;
		
		case 12:
		name = "nu_e";
		break;
		
		case -12:
		name = "nubar_e";
		break;
		
		case 13:
		name = "mu-";
		break;
		
		case -13:
		name = "mu+";
		break;
		
		case 14:
		name = "nu_mu";
		break;
		
		case -14:
		name = "nubar_mu";
		break;
		
		case 15:
		name = "tau-";
		break;
		
		case -15:
		name = "tau+";
		break;
		
		case 16:
		name = "nu_tau";
		break;
		
		case -16:
		name = "nubar_tau";
		break;
		
		case 17:
		name = "tau'";
		break;
		
		case -17:
		name = "taubar'";
		break;
		
		case 18:
		name = "nu'_tau";
		break;
		
		case -18:
		name = "nubar'_tau";
		break;
		
		// Gauge boson and other fundamental boson codes
		
		case 21:
		name = "g";
		break;
		
		case 22:
		name = "gamma";
		break;
		
		case 23:
		name = "Z0";
		break;
		
		case 24:
		name = "W+";
		break;
		
		case -24:
		name = "W-";
		break;
		
		case 25:
		name = "h0";
		break;
		
		case 32:
		name = "Z'0";
		break;
		
		case 33:
		name = "Z''0";
		break;
		
		case 34:
		name = "W'+";
		break;
		
		case -34:
		name = "W'-";
		break;
		
		case 35:
		name = "H0";
		break;
		
		case 36:
		name = "A0";
		break;
		
		case 37:
		name = "H+";
		break;
		
		case -37:
		name = "H-";
		break;
		
		case 39:
		name = "G";
		break;
		
		case 41:
		name = "R0";
		break;
		
		case 42:
		name = "LQ";
		break;
		
		// Various special codes
		
		case 81:
		name = "specflav";
		break;
		
		case 82:
		name = "rndmflav";
		break;
		
		case 83:
		name = "phasespa";
		break;
		
		case 84:
		name = "c-hadron";
		break;
		
		case 85:
		name = "b-hadron";
		break;
		
		case 88:
		name = "junction";
		break;
		
		case 90:
		name = "system";
		break;
		
		case 91:
		name = "cluster";
		break;
		
		case 92:
		name = "string";
		break;
		
		case 93:
		name = "indep.";
		break;
		
		case 94:
		name = "CMshower";
		break;
		
		case 95:
		name = "SPHEaxis";
		break;
		
		case 96:
		name = "THRUaxis";
		break;
		
		case 97:
		name = "CLUSjet";
		break;
		
		case 98:
		name = "CELLjet";
		break;
		
		case 99:
		name = "table";
		break;
		
		// Diquark codes
		
		case 2101:
		name = "ud_0";
		break;
		
		case 3101:
		name = "sd_0";
		break;
		
		case 3201:
		name = "su_0";
		break;
		
		case 1103:
		name = "dd_1";
		break;
		
		case 2103:
		name = "ud_1";
		break;
		
		case 2203:
		name = "uu_1";
		break;
		
		case 3103:
		name = "sd_1";
		break;
		
		case 3203:
		name = "su_1";
		break;
		
		case 3303:
		name = "ss_1";
		break;
		
		// Meson codes
		
		case 211:
		name = "pi+";
		break;
		
		case -211:
		name = "pi-";
		break;
		
		case 311:
		name = "K0";
		break;
		
		case -311:
		name = "K0bar";
		break;
		
		case 321:
		name = "K+";
		break;
		
		case -321:
		name = "K-";
		break;
		
		case 411:
		name = "D+";
		break;
		
		case -411:
		name = "D-";
		break;
		
		case 421:
		name = "D0";
		break;
		
		case -421:
		name = "D0bar";
		break;
		
		case 431:
		name = "D_s+";
		break;
		
		case -431:
		name = "D_s-";
		break;
		
		case 511:
		name = "B0";
		break;
		
		case -511:
		name = "B0bar";
		break;
		
		case 521:
		name = "B+";
		break;
		
		case -521:
		name = "B-";
		break;
		
		case 531:
		name = "B_s0";
		break;
		
		case 541:
		name = "B_c+";
		break;
		
		case -541:
		name = "B_c-";
		break;
		
		case 111:
		name = "pi0";
		break;
		
		case 221:
		name = "eta";
		break;
		
		case 331:
		name = "eta'";
		break;
		
		case 441:
		name = "eta_c";
		break;
		
		case 551:
		name = "eta_b";
		break;
		
		case 130:
		name = "K_L0";
		break;
		
		case 310:
		name = "K_S0";
		break;
		
		case 213:
		name = "rho+";
		break;
		
		case -213:
		name = "rho-";
		break;
		
		case 313:
		name = "K*0";
		break;
		
		case 323:
		name = "K*+";
		break;
		
		case -323:
		name = "K*-";
		break;
		
		case 413:
		name = "D*+";
		break;
		
		case -413:
		name = "D*-";
		break;
		
		case 423:
		name = "D*0";
		break;
		
		case 433:
		name = "D*_s+";
		break;
		
		case -433:
		name = "D*_s-";
		break;
		
		case 513:
		name = "B*0";
		break;
		
		case -513:
		name = "B*0bar";
		break;
		
		case 523:
		name = "B*+";
		break;
		
		case -523:
		name = "B*-";
		break;
		
		case 533:
		name = "B*_s0";
		break;
		
		case 543:
		name = "B*_c+";
		break;
		
		case 113:
		name = "rho0";
		break;
		
		case 223:
		name = "omega";
		break;
		
		case 333:
		name = "phi";
		break;
		
		case 443:
		name = "J/psi";
		break;
		
		case 553:
		name = "Upsilon";
		break;
		
		// Baryon codes
		
		case 2112:
		name = "n0";
		break;
		
		case 2212:
		name = "p+";
		break;
		
		case -2212:
		name = "p-";
		break;
		
		case 3112:
		name = "Sigma-";
		break;
		
		case -3112:
		name = "Sigma+";
		break;
		
		case 3122:
		name = "Lambda0";
		break;
		
		case 3212:
		name = "Sigma0";
		break;
		
		case 3222:
		name = "Sigma+";
		break;
		
		case -3222:
		name = "Sigma-";
		break;
		
		case 3312:
		name = "Xi-";
		break;
		
		case -3312:
		name = "Xi+";
		break;
		
		case 3322:
		name = "Xi0";
		break;
		
		case 4112:
		name = "Sigma_c0";
		break;
		
		case 4122:
		name = "Lambda_c+";
		break;
		
		case -4122:
		name = "Lambda_c-";
		break;
		
		case 4212:
		name = "Sigma_c+";
		break;
		
		case -4212:
		name = "Sigma_c-";
		break;
		
		case 4222:
		name = "Sigma_c++";
		break;
		
		case -4222:
		name = "Sigma_c--";
		break;
		
		case 4132:
		name = "Xi_c0";
		break;
		
		case 4312:
		name = "Xi'_c0";
		break;
		
		case 4232:
		name = "Xi_c+";
		break;
		
		case -4232:
		name = "Xi_c-";
		break;
		
		case 4322:
		name = "Xi'_c+";
		break;
		
		case -4322:
		name = "Xi'_c-";
		break;
		
		case 4332:
		name = "Omega_c0";
		break;
		
		case 5112:
		name = "Sigma_b-";
		break;
		
		case 5122:
		name = "Lambda_b0";
		break;
		
		case 5212:
		name = "Sigma_b0";
		break;
		
		case 5222:
		name = "Sigma_b+";
		break;
		
		case 1114:
		name = "Delta-";
		break;
		
		case 2214:
		name = "Delta+";
		break;
		
		case 2114:
		name = "Delta0";
		break;
		
		case 2224:
		name = "Delta++";
		break;
		
		case 3114:
		name = "Sigma*-";
		break;
		
		case 3214:
		name = "Sigma*0";
		break;
		
		case 3224:
		name = "Sigma*+";
		break;
		
		case 3314:
		name = "Xi*-";
		break;
		
		case -3314:
		name = "Xi*+";
		break;
		
		case 3324:
		name = "Xi*0";
		break;
		
		case 3334:
		name = "Omega-";
		break;
		
		case -3334:
		name = "Omega+";
		break;
		
		case 4114:
		name = "Sigma*_c0";
		break;
		
		case 4214:
		name = "Sigma*_c+";
		break;
		
		case -4214:
		name = "Sigma*_c-";
		break;
		
		case 4224:
		name = "Sigma*_c++";
		break;
		
		case -4224:
		name = "Sigma*_c--";
		break;
		
		case 4314:
		name = "Xi*_c0";
		break;
		
		case 4324:
		name = "Xi*_c+";
		break;
		
		case -4324:
		name = "Xi*_c-";
		break;
		
		case 4334:
		name = "Omega*_c0";
		break;
		
		case 5114:
		name = "Sigma*_b-";
		break;
		
		case 5224:
		name = "Sigma*_b+";
		break;
		
		case 5214:
		name = "Sigma*_b0";
		break;
		
		// QCD effective states
		
		case 110:
		name = "reggeon";
		break;
		
		case 990:
		name = "pomeron";
		break;
		
		case 9900110:
		name = "rho_diff0";
		break;
		
		case 9900210:
		name = "pi_diffr+";
		break;
		
		case 9900220:
		name = "omega_di0";
		break;
		
		case 9900330:
		name = "phi_diff0";
		break;
		
		case 9900440:
		name = "J/psi_di0";
		break;
		
		case 9902110:
		name = "n_diffr";
		break;
		
		case 9902210:
		name = "p_diffr+";
		break;
		
		// Supersymmetric codes - Left squarks

		case 1000001:
		name = "~d_L";
		break;
		
		case -1000001:
		name = "~dbar_L";
		break;
		
		case 1000002:
		name = "~u_L";
		break;
		
		case -1000002:
		name = "~ubar_L";
		break;
		
		case 1000003:
		name = "~s_L";
		break;
		
		case -1000003:
		name = "~sbar_L";
		break;
		
		case 1000004:
		name = "~c_L";
		break;
		
		case -1000004:
		name = "~cbar_L";
		break;
		
		case 1000005:
		name = "~b_1";
		break;
		
		case -1000005:
		name = "~bbar_1";
		break;
		
		case 1000006:
		name = "~t_1";
		break;
		
		case -1000006:
		name = "~tbar_1";
		break;
		
		// Supersymmetric codes - Left sleptons
		
		case 1000011:
		name = "~e_L-";
		break;
		
		case -1000011:
		name = "~e_L+";
		break;
		
		case 1000012:
		name = "~nu_eL";
		break;
		
		case -1000012:
		name = "~nubar_eL";
		break;
		
		case 1000013:
		name = "~mu_L-";
		break;
		
		case -1000013:
		name = "~mu_L+";
		break;
		
		case 1000014:
		name = "~nu_muL";
		break;
		
		case -1000014:
		name = "~nubar_muL";
		break;
		
		case 1000015:
		name = "~tau_L-";
		break;
		
		case -1000015:
		name = "~tau_L+";
		break;
		
		case 1000016:
		name = "~nu_tauL";
		break;
		
		case -1000016:
		name = "~nubar_tauL";
		break;
		
		// Supersymmetric codes - Right squarks

		case 2000001:
		name = "~d_R";
		break;
		
		case -2000001:
		name = "~dbar_R";
		break;
		
		case 2000002:
		name = "~u_R";
		break;
		
		case -2000002:
		name = "~ubar_R";
		break;
		
		case 2000003:
		name = "~s_R";
		break;
		
		case -2000003:
		name = "~sbar_R";
		break;
		
		case 2000004:
		name = "~c_R";
		break;
		
		case -2000004:
		name = "~cbar_R";
		break;
		
		case 2000005:
		name = "~b_2";
		break;
		
		case -2000005:
		name = "~bbar_2";
		break;
		
		case 2000006:
		name = "~t_2";
		break;
		
		case -2000006:
		name = "~tbar_2";
		break;
		
		// Supersymmetric codes - Right sleptons
		
		case 2000011:
		name = "~e_R-";
		break;
		
		case -2000011:
		name = "~e_R+";
		break;
		
		case 2000012:
		name = "~nu_eR";
		break;
		
		case -2000012:
		name = "~nubar_eR";
		break;
		
		case 2000013:
		name = "~mu_R-";
		break;
		
		case -2000013:
		name = "~mu_R+";
		break;
		
		case 2000014:
		name = "~nu_muR";
		break;
		
		case -2000014:
		name = "~nubar_muR";
		break;
		
		case 2000015:
		name = "~tau_R-";
		break;
		
		case -2000015:
		name = "~tau_R+";
		break;
		
		case 2000016:
		name = "~nu_tauR";
		break;
		
		case -2000016:
		name = "~nubar_tauR";
		break;
		
		// Supersymmetric codes - gauginos
		
		case 1000021:
		name = "~g";
		break;
		
		case 1000022:
		name = "~chi_10";
		break;
		
		case 1000023:
		name = "~chi_20";
		break;
		
		case 1000024:
		name = "~chi_1+";
		break;
		
		case -1000024:
		name = "~chi_1-";
		break;
		
		case 1000025:
		name = "~chi_30";
		break;
		
		case 1000035:
		name = "~chi_40";
		break;
		
		case 1000037:
		name = "~chi_2+";
		break;
		
		case -1000037:
		name = "~chi_2-";
		break;
		
		case 1000039:
		name = "~Gravitino";
		break;
		
		case 1000045:
		name = "~chi_50";
		break;
		
		case 45:
		name = "H_30";
		break;
		
		case 46:
		name = "A_20";
		break;

	}
	
	return name;
}

#endif // #ifdef PdgIdConverter_cxx
