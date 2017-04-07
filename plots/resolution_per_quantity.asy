import root;
import pad_layout;

string topDir = "../";

string n_events = "1E5";
string simulations[], simu_labels[];
simulations.push("simulations/"+n_events+"/vtx_y,ang,xi.root"); simu_labels.push("$y^*$, $\th^*_x$, $\th_y^*$, $\xi$");
simulations.push("simulations/"+n_events+"/vtx,ang,xi.root"); simu_labels.push("+ $x^*$");
simulations.push("simulations/"+n_events+"/vtx,ang,xi,det.root"); simu_labels.push("+ det.~res.");
simulations.push("simulations/"+n_events+"/vtx,ang,xi,det,bd.root"); simu_labels.push("+ beam div.");

string sectors[] = {
	"45",
	"56",
};

string quantities[], q_labels[], q_units[], q_fmts[];
real q_scales[];

quantities.push("vtx_x"); q_labels.push("x^*"); q_scales.push(1e6); q_units.push("\mu m"); q_fmts.push("%#.1f");
quantities.push("vtx_y"); q_labels.push("y^*"); q_scales.push(1e6); q_units.push("\mu m"); q_fmts.push("%#.1f");
quantities.push("th_x"); q_labels.push("\th_x^*"); q_scales.push(1e6); q_units.push("\mu rad"); q_fmts.push("%#.1f");
quantities.push("th_y"); q_labels.push("\th_y^*"); q_scales.push(1e6); q_units.push("\mu rad"); q_fmts.push("%#.1f");
quantities.push("xi"); q_labels.push("\xi"); q_scales.push(1); q_units.push(""); q_fmts.push("%#.4f");

TGraph_errorBar = None;

//----------------------------------------------------------------------------------------------------

void PlotAllHist(string f, string hist, real x_scale=1, string fmt = "%#.1f")
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

void PlotAllGraph(string f, string hist, real x_scale=1, real y_scale)
{
	for (int seci : sectors.keys)
	{
		string sector = sectors[seci];
		RootObject obj = RootGetObject(f, "sector " + sector + "/" + hist + "_" + sector);
		pen p = StdPen(seci + 1);
		draw(scale(x_scale, y_scale), obj, "l,p", p, mCi+2pt+p, "sector " + sector);
	}
}

//----------------------------------------------------------------------------------------------------

for (int qi : quantities.keys)
{
	string quantity = quantities[qi];

	NewRow();

	for (int simi : simulations.keys)
	{
		NewPad(false);
		label("{\SetFontSizesXX " + simu_labels[simi] + "}");
	}

	NewRow();

	for (string simulation : simulations)
	{
		string f = topDir + simulation;
	
		string axisLabel = "\De " + q_labels[qi];
		if (q_units[qi] != "")
			axisLabel += "\ung{" + q_units[qi] + "}";
		NewPad("$" + axisLabel + "$");
		PlotAllHist(f, "h_de_" + quantity, q_scales[qi], q_fmts[qi]);
		AttachLegend(BuildLegend(lineLength=5mm));
	}

	NewRow();

	for (string simulation : simulations)
	{
		string f = topDir + simulation;
	
		string axisLabel = "\De " + q_labels[qi];
		if (q_units[qi] != "")
			axisLabel += "\ung{" + q_units[qi] + "}";
		NewPad("$\xi$", "RMS of $" + axisLabel + "$");
		PlotAllGraph(f, "g_rms_de_" + quantity + "_vs_xi", 1., q_scales[qi]);
		AttachLegend(BuildLegend(lineLength=5mm));
	}

	GShipout("resolution_per_quantity_" + quantity, vSkip=1mm);

	break;
}
