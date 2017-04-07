import root;
import pad_layout;

string rp_tags[], rp_labels[];
rp_tags.push("3"); rp_labels.push("45-210-fr");
rp_tags.push("2"); rp_labels.push("45-210-nr");
rp_tags.push("102"); rp_labels.push("56-210-nr");
rp_tags.push("103"); rp_labels.push("56-210-fr");

string f = "../get_optical_functions.root";

//----------------------------------------------------------------------------------------------------

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	NewPad("$\xi$", "$x_0\ung{mm}$");
	draw(scale(1., 1e3), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_x0_vs_xi"), red);

	//--------------------

	NewPad("$\xi$", "$y_0\ung{mm}$");
	draw(scale(1., 1e3), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_y0_vs_xi"), red);

	//--------------------

	NewPad("$\xi$", "$v_x$");
	draw(scale(1., 1e0), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_v_x_vs_xi"), red);

	//--------------------

	NewPad("$\xi$", "$v_y$");
	draw(scale(1., 1e0), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_v_y_vs_xi"), red);

	//--------------------

	NewPad("$\xi$", "$L_x\ung{mm}$");
	draw(scale(1., 1e0), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_L_x_vs_xi"), red);

	//--------------------

	NewPad("$\xi$", "$L_y\ung{mm}$");
	draw(scale(1., 1e0), RootGetObject(f, "RP"+rp_tags[rpi]+"/g_L_y_vs_xi"), red);
}
