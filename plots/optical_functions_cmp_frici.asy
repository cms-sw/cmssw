import root;
import pad_layout;

string rp_tags[], rp_labels[], rp_frici_file[], rp_frici_obj[];
rp_tags.push("3"); rp_labels.push("45-210-fr"); rp_frici_file.push("xi_as_a_function_of_x_graph_b2.root"); rp_frici_obj.push("XRPH_D6L5_B2");
rp_tags.push("2"); rp_labels.push("45-210-nr"); rp_frici_file.push("xi_as_a_function_of_x_graph_b2.root"); rp_frici_obj.push("XRPH_C6L5_B2");
rp_tags.push("102"); rp_labels.push("56-210-nr"); rp_frici_file.push("xi_as_a_function_of_x_graph_b1.root"); rp_frici_obj.push("XRPH_C6R5_B1");
rp_tags.push("103"); rp_labels.push("56-210-fr"); rp_frici_file.push("xi_as_a_function_of_x_graph_b1.root"); rp_frici_obj.push("XRPH_D6R5_B1");

string files[], f_labels[];
pen f_pens[];
//files.push("../get_optical_functions_0E-6.root"); f_labels.push("$y^*_0 = 0\un{\mu m}$"); f_pens.push(black);
//files.push("../get_optical_functions_250E-6.root"); f_labels.push("$y^*_0 = 250\un{\mu m}$"); f_pens.push(blue);
files.push("../get_optical_functions_550E-6.root"); f_labels.push("$y^*_0 = 550\un{\mu m}$"); f_pens.push(red);

//----------------------------------------------------------------------------------------------------

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	NewPad("$x - x(\xi = 0)\ung{mm}$", "$\xi$");

	for (int fi : files.keys)
		draw(scale(1e3, 1.), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_xi_vs_xso"), f_pens[fi]);

	string frici_dir = "/afs/cern.ch/work/j/jkaspar/software/offline/704/user/ctpps_optics/test/frici/";

	draw(scale(1e3, -1), RootGetObject(frici_dir + rp_frici_file[rpi], rp_frici_obj[rpi]), black+1.5pt+dashed);	

	limits((0, 0), (15, 0.15), Crop);
	

	//--------------------

	NewPad(false);

	AddToLegend("<optics parametrisation:");
	for (int fi : files.keys)
		AddToLegend(f_labels[fi], f_pens[fi]);	

	AddToLegend("<curves by Frici:");
	AddToLegend("x-to-xi", black+1.5pt+dashed);

	AttachLegend();
}
