import root;
import pad_layout;

string rp_tags[], rp_labels[], rps[];
rp_tags.push("3"); rp_labels.push("45-210-fr"); rps.push("L_1_F");
rp_tags.push("2"); rp_labels.push("45-210-nr"); rps.push("L_1_N");
//rp_tags.push("102"); rp_labels.push("56-210-nr"); rps.push("R_1_N");
//rp_tags.push("103"); rp_labels.push("56-210-fr"); rps.push("R_1_F");


string vtx_y_tags[], vtx_y_labels[];
vtx_y_tags.push("0E-6"); vtx_y_labels.push("$y_0^* = 0\un{\mu m}$");
vtx_y_tags.push("100E-6"); vtx_y_labels.push("$y_0^* = 100\un{\mu m}$");
vtx_y_tags.push("200E-6"); vtx_y_labels.push("$y_0^* = 200\un{\mu m}$");
vtx_y_tags.push("300E-6"); vtx_y_labels.push("$y_0^* = 300\un{\mu m}$");
vtx_y_tags.push("400E-6"); vtx_y_labels.push("$y_0^* = 400\un{\mu m}$");

string dir_data = "/afs/cern.ch/work/j/jkaspar/analyses/ctpps/alignment/";

string data_fills[], data_files[];
real data_offsets_45[], data_offsets_56[];
real offset_45 = -0.455, offset_56 = -0.05;
data_fills.push("4988"); data_files.push(dir_data + "period1_physics/fill_4988/y_alignment.root"); data_offsets_45.push(offset_45-0.); data_offsets_56.push(offset_56-0.400);
data_fills.push("5026"); data_files.push(dir_data + "period1_physics/fill_5026/y_alignment.root"); data_offsets_45.push(offset_45-0.1); data_offsets_56.push(offset_56-0.420);
data_fills.push("5266"); data_files.push(dir_data + "period1_physics/fill_5266/y_alignment.root"); data_offsets_45.push(offset_45+0.01); data_offsets_56.push(offset_56-0.365);


//----------------------------------------------------------------------------------------------------

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	NewPad("$x - x(\xi = 0) \ung{mm}$", "$y - y(\xi = 0)\ung{mm}$");

	AddToLegend("<data:");

	for (int fi : data_fills.keys)
	{
		TH1_x_min = 4;
		TH1_x_max = 12;
		RootObject profile = RootGetObject(data_files[fi], rps[rpi] + "/p_y_vs_x");

		real offset = 0.;

		if (rp_tags[rpi] == "2" || rp_tags[rpi] == "3")
			offset = -data_offsets_45[fi];

		if (rp_tags[rpi] == "102" || rp_tags[rpi] == "103")
			offset = -data_offsets_56[fi];

		if (rp_tags[rpi] == "2")
			offset -= 0.13;

		if (rp_tags[rpi] == "102")
			offset -= 0.09;

		draw(shift(0, offset), profile, "eb", StdPen(fi));
		AddToLegend("fill " + data_fills[fi], mPl+4pt+StdPen(fi));
	}

	AddToLegend("<optics parametrisation:");
	
	for (int vi : vtx_y_tags.keys)
	{
		string f = "../get_optical_functions_"+vtx_y_tags[vi]+".root";
		draw(scale(1e3, 1e3), RootGetObject(f, "RP" + rp_tags[rpi] + "/g_y0so_vs_x0so"), StdPen(vi+1)+1pt, vtx_y_labels[vi]);
	}

	limits((0, -0.5), (12, 0.1), Crop);

	frame f_legend = BuildLegend();

	NewPad(false);
	add(shift(0, 80) * f_legend);
}
