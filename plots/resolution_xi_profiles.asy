import root;
import pad_layout;

string topDir = "../";

string n_events = "1E4";
string simulations[] = {
	"simulations/"+n_events+"/vtx_y,ang,xi.root",
	"simulations/"+n_events+"/vtx,ang,xi.root",
	"simulations/"+n_events+"/vtx,ang,xi,det.root",
	"simulations/"+n_events+"/vtx,ang,xi,det,bd.root",
};

string sectors[] = {
	"45",
	"56",
};


TGraph_errorBar = None;

//----------------------------------------------------------------------------------------------------

void PlotAll(string f, string hist, real x_scale=1, real y_scale)
{
	for (int seci : sectors.keys)
	{
		string sector = sectors[seci];
		RootObject obj = RootGetObject(f, "sector " + sector + "/" + hist + "_" + sector);
		pen p = StdPen(seci + 1);
		draw(scale(x_scale, y_scale), obj, "d0,p", p, mCi+2pt+p, "sector " + sector);
	}
}

//----------------------------------------------------------------------------------------------------

for (string simulation : simulations)
{
	NewRow();

	NewPad(false);
	label(replace("\vbox{\SetFontSizesXX\hbox{" + simulation + "}}", "_", "\_"));

	string f = topDir + simulation;

	//--------------------

	NewPad("$\xi$", "RMS of $\De x^{*}\ung{\mu m}$");
	PlotAll(f, "g_rms_de_vtx_x_vs_xi", 1., 1e6);
	xlimits(0, 0.2, Crop);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\xi$", "RMS of $\De y^{*}\ung{\mu m}$");
	PlotAll(f, "g_rms_de_vtx_y_vs_xi", 1., 1e6);
	xlimits(0, 0.2, Crop);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\xi$", "RMS of $\De \th_x^{*}\ung{\mu rad}$");
	PlotAll(f, "g_rms_de_th_x_vs_xi", 1., 1e6);
	xlimits(0, 0.2, Crop);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\xi$", "RMS of $\De \th_y^{*}\ung{\mu rad}$");
	PlotAll(f, "g_rms_de_th_y_vs_xi", 1., 1e6);
	xlimits(0, 0.2, Crop);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\xi$", "RMS of $\De\xi$");
	PlotAll(f, "g_rms_de_xi_vs_xi", 1., 1.);
	xlimits(0, 0.2, Crop);
	AttachLegend(BuildLegend(lineLength=5mm));
}
