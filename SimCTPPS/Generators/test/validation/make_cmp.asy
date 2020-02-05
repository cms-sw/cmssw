import root;
import pad_layout;

string src_dir = "../output/version11-test/Z/";
string ref_dir = "/afs/cern.ch/work/j/jkaspar/work/software/ctpps/development/proton_reco_step4/CMSSW_10_6_1/src/SimCTPPS/Generators/test/output/version9/Z/";

string cfg_masses[], cfg_xangles[], cfg_periods[];
cfg_masses.push("1200"); cfg_xangles.push("120"); cfg_periods.push("2017_preTS2");
cfg_masses.push("1200"); cfg_xangles.push("150"); cfg_periods.push("2017_preTS2");
cfg_masses.push("1200"); cfg_xangles.push("120"); cfg_periods.push("2017_postTS2");
cfg_masses.push("1200"); cfg_xangles.push("150"); cfg_periods.push("2017_postTS2");

string hists[];
hists.push("h_p_z_LAB_2p");

TH1_x_min = -1000;
TH1_x_max = +1000;

//----------------------------------------------------------------------------------------------------

for (int cfgi : cfg_masses.keys)
{
	NewRow();

	NewPadLabel(replace(cfg_masses[cfgi] + ", " + cfg_xangles[cfgi] + ", " + cfg_periods[cfgi], "_", "\_"));

	//string f_src = src_dir + "ppxzGeneratorValidation_" + cfg_masses[cfgi] + "_" + cfg_xangles[cfgi] + "_" + cfg_periods[cfgi] + ".root";
	string f_src = src_dir + "m_X_" + cfg_masses[cfgi] + "/xangle_" + cfg_xangles[cfgi] + "/" + cfg_periods[cfgi] + "/ppxzGeneratorValidation.root";
	string f_ref = ref_dir + "m_X_" + cfg_masses[cfgi] + "/xangle_" + cfg_xangles[cfgi] + "/" + cfg_periods[cfgi] + "/ppxzGeneratorValidation.root";

	for (int hi : hists.keys)
	{
		NewPad();

		string pth = "after simulation/" + hists[hi];

		draw(RootGetObject(f_src, pth), "vl", blue);
		draw(RootGetObject(f_ref, pth), "vl", red+dashed);
	}
}
