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

//----------------------------------------------------------------------------------------------------

void PlotAll(string f, string hist, real x_scale=1, string fmt = "%#.1f")
{
	for (int seci : sectors.keys)
	{
		string sector = sectors[seci];
		RootObject obj = RootGetObject(f, "sector " + sector + "/" + hist + "_" + sector);
		draw(scale(x_scale, 1.), obj, "vl", StdPen(seci + 1), "sector " + sector);
		AddToLegend(format("RMS = $" + fmt + "$", obj.rExec("GetRMS") * x_scale));
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

	NewPad("$x^{*,\rm reco} - x^{*,\rm sim}\ung{\mu m}$");
	PlotAll(f, "h_de_vtx_x", 1e6);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$y^{*,\rm reco} - y^{*,\rm sim}\ung{\mu m}$");
	PlotAll(f, "h_de_vtx_y", 1e6);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\th_x^{*,\rm reco} - \th_x^{*,\rm sim}\ung{\mu rad}$");
	PlotAll(f, "h_de_th_x", 1e6);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\th_y^{*,\rm reco} - \th_y^{*,\rm sim}\ung{\mu rad}$");
	PlotAll(f, "h_de_th_y", 1e6);
	AttachLegend(BuildLegend(lineLength=5mm));

	//--------------------

	NewPad("$\xi^{\rm reco} - \xi^{\rm sim}$");
	PlotAll(f, "h_de_xi", 1, "%#.4f");
	AttachLegend(BuildLegend(lineLength=5mm));
}
