import root;
import pad_layout;

//----------------------------------------------------------------------------------------------------


frame f_legend;

void CompareOne(string tag, string f_file, string f_obj, string rpIdTag)
{
	string p_file = "../../ctpps_optics/test/test.root";

	NewPad("x\ung{mm}", "$\xi$");

	//AddToLegend("<Frici, 22 Sep 2016:");
	//draw(RootGetObject(f_file, f_obj), blue, "offset removed");

	AddToLegend("<parametrisation, vesion 3 (26 Nov 2016):");
	string p_dir = tag;
	draw(scale(1e3, -1), RootGetObject(p_file, "version3/"+p_dir+"/g_xi_vs_x"), magenta+1pt, "validation");

	string simu_file = "../get_optical_functions.root";
	draw(scale(1e3, 1), RootGetObject(simu_file, "RP" + rpIdTag + "/g_xi_vs_x"), "l", black+dashed+1.5pt);

	f_legend = BuildLegend();

	currentpicture.legend.delete();
	AttachLegend(tag);
}

//----------------------------------------------------------------------------------------------------

CompareOne("L-210-nr-hr", "frici/xi_as_a_function_of_x_graph_b2.root", "XRPH_C6L5_B2", "2");
CompareOne("L-210-fr-hr", "frici/xi_as_a_function_of_x_graph_b2.root", "XRPH_D6L5_B2", "3");

NewRow();

CompareOne("R-210-nr-hr", "frici/xi_as_a_function_of_x_graph_b1.root", "XRPH_C6R5_B1", "102");
CompareOne("R-210-fr-hr", "frici/xi_as_a_function_of_x_graph_b1.root", "XRPH_D6R5_B1", "103");

NewPad(false);
attach(f_legend);
