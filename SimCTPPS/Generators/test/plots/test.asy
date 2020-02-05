import root;
import pad_layout;

string topDir = "../";

string files[], f_labels[];
pen f_pens[];

/*
files.push("output/version6/Z/m_X_1200/xangle_140/2017_postTS2"); f_pens.push(black); f_labels.push("[version6] simu: v4, reco: v4");
files.push("output/version7/Z/m_X_1200/xangle_140/2017_postTS2"); f_pens.push(green+dashed); f_labels.push("[version7] simu: v4, reco: v4");
files.push("output/version7/Z/m_X_1200/xangle_140/2017_postTS2_opt_v1_simu"); f_pens.push(red); f_labels.push("[version7] simu: v1, reco: v4");
files.push("output/version7/Z/m_X_1200/xangle_140/2017_postTS2_opt_v1_simu_reco"); f_pens.push(blue); f_labels.push("[version7] simu: v1, reco: v1");
*/

//files.push("output/version6/Z/m_X_1200/xangle_140/2017_postTS2"); f_pens.push(black); f_labels.push("[version6]");
files.push("output/version7/Z/m_X_1200/xangle_120/2017_postTS2"); f_pens.push(blue); f_labels.push("[version7]");
//files.push("output/version8/Z/m_X_1200/xangle_140/2017_postTS2"); f_pens.push(blue); f_labels.push("[version8]");
files.push("output/version9/Z/m_X_1200/xangle_120/2017_postTS2"); f_pens.push(red); f_labels.push("[version9]");

bool rebin = false;

xSizeDef = 12cm;

//----------------------------------------------------------------------------------------------------

NewPad(false);

for (int fi : files.keys)
{
	AddToLegend(f_labels[fi], f_pens[fi]);
}

AttachLegend();

//---------------------------------------------------------------------------------------------------
NewRow();

TH1_x_min = 1;
TH1_x_max = 17;

NewPad("$x\ung{mm}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/output_shape_smear.root", "RP 3/h_x");
	if (rebin)
		obj.vExec("Rebin", 2);

	draw(obj, "vl", f_pens[fi]);
}

//yaxis(XEquals(3.40, false), heavygreen);

AttachLegend("RP 3");

//--------------------------------------------------

NewPad("$x\ung{mm}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/output_shape_smear.root", "RP 103/h_x");
	if (rebin)
		obj.vExec("Rebin", 2);

	draw(obj, "vl", f_pens[fi]);
}

//yaxis(XEquals(2.41, false), heavygreen);

AttachLegend("RP 103");

//---------------------------------------------------------------------------------------------------
NewRow();

NewPad("$x\ung{mm}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/output_shape_smear.root", "RP 23/h_x");
	if (rebin)
		obj.vExec("Rebin", 2);

	draw(obj, "vl", f_pens[fi]);
}

//yaxis(XEquals(3.44, false), heavygreen);

AttachLegend("RP 23");

//--------------------------------------------------

NewPad("$x\ung{mm}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/output_shape_smear.root", "RP 123/h_x");
	if (rebin)
		obj.vExec("Rebin", 2);

	draw(obj, "vl", f_pens[fi]);
}

//yaxis(XEquals(2.37, false), heavygreen);

AttachLegend("RP 123");

/*

//----------------------------------------------------------------------------------------------------
NewRow();

TH1_x_max = +inf;

NewPad("$\De m_X\ung{GeV}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_m_X_single");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("single-RP");

//--------------------------------------------------

NewPad("$\De m_X\ung{GeV}$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_m_X_multi");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("multi-RP");

//----------------------------------------------------------------------------------------------------
NewRow();

TH1_x_max = +inf;

NewPad("$\De\xi$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_xi_single_45");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("single-RP, sec 45");

//--------------------------------------------------

NewPad("$\De\xi$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_xi_multi_45");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("multi-RP, sec 45");

//----------------------------------------------------------------------------------------------------
NewRow();

TH1_x_max = +inf;

NewPad("$\De\xi$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_xi_single_56");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("single-RP, sec 56");

//--------------------------------------------------

NewPad("$\De\xi$");

for (int fi : files.keys)
{
	RootObject obj = RootGetObject(topDir + files[fi] + "/ppxzGeneratorValidation.root", "after simulation/h_de_xi_multi_56");
	if (rebin)
		obj.vExec("Rebin", 4);

	draw(obj, "vl", f_pens[fi]);
}

AttachLegend("multi-RP, sec 56");

*/
