import root;
import pad_layout;

string rp_tags[], rp_labels[];
rp_tags.push("3"); rp_labels.push("45-210-fr");
rp_tags.push("2"); rp_labels.push("45-210-nr");
rp_tags.push("102"); rp_labels.push("56-210-nr");
rp_tags.push("103"); rp_labels.push("56-210-fr");

string files[], f_labels[];
pen f_pens[];
files.push("../get_optical_functions_0E-6.root"); f_labels.push("$y^*_0 = 0\un{\mu m}$"); f_pens.push(black);
files.push("../get_optical_functions_250E-6.root"); f_labels.push("$y^*_0 = 250\un{\mu m}$"); f_pens.push(blue);
files.push("../get_optical_functions_550E-6.root"); f_labels.push("$y^*_0 = 550\un{\mu m}$"); f_pens.push(red);

//----------------------------------------------------------------------------------------------------

struct OptFunRecord
{
	string rpId;
	string desc;
	mark m;
	real xi;
	real x, y, px, py, dx, dy, vx, Lx, vy, Ly;
};

OptFunRecord OptFunRecord(string _rpId, string _desc, mark _m, real _xi, real _x, real _y, real _px, real _py, real _dx, real _dy, real _vx, real _Lx, real _vy, real _Ly)
{
	OptFunRecord r;
	r.rpId = _rpId;
	r.desc = _desc;
	r.m = _m;
	r.xi = _xi;
	r.x = _x;
	r.y = _y;
	r.px = _px;
	r.py = _py;
	r.dx = _dx;
	r.dy = _dy;
	r.vx = _vx;
	r.Lx = _Lx;
	r.vy = _vy;
	r.Ly = _Ly;

	return r;
}

OptFunRecord optFunRecords[];

// ----- beam 1 -----

// y0 = 0
optFunRecords.push(OptFunRecord("102", "twiss/y0=0", mTU+4pt+black, 0., 2.93039e-3, 1.30854e-3, -4.97971e-5, 1.22081e-5, -6.95183e-2, -1.13725e-3, -3.97470e+0, 6.53173e+0, -2.99717e+0, 1.63656e+1));
optFunRecords.push(OptFunRecord("103", "twiss/y0=0", mTU+4pt+black, 0., 2.49591e-3, 1.41506e-3, -4.97971e-5, 1.22081e-5, -6.57844e-2, -1.20079e-3, -4.05770e+0, 4.47299e+0, -3.22849e+0, 1.47176e+1));

optFunRecords.push(OptFunRecord("102", "twiss/y0=0", mTU+4pt+black, 0.05, 6.57927e-3, 1.35897e-3, -7.07638e-5, 1.25057e-5, -7.22646e-2, -1.14966e-3, -4.18582e+0, -9.44711e+0, -3.52318e+0, -3.24459e+0));
optFunRecords.push(OptFunRecord("103", "twiss/y0=0", mTU+4pt+black, 0.05, 5.96185e-3, 1.46808e-3, -7.07638e-5, 1.25057e-5, -6.80304e-2, -1.20432e-3, -4.25765e+0, -1.16936e+1, -3.77651e+0, -5.95435e+0));

optFunRecords.push(OptFunRecord("102", "twiss/y0=0", mTU+4pt+black, 0.10, 1.07722e-2, 1.41251e-3, -9.61175e-5, 1.27546e-5, -7.59215e-2, -1.15567e-3, -4.38183e+0, -2.71793e+1, -4.13329e+0, -2.57216e+1));
optFunRecords.push(OptFunRecord("103", "twiss/y0=0", mTU+4pt+black, 0.10, 9.93355e-3, 1.52379e-3, -9.61175e-5, 1.27546e-5, -7.10779e-2, -1.19765e-3, -4.43902e+0, -2.95252e+1, -4.40846e+0, -2.95449e+1));


// y0 = 250 um
optFunRecords.push(OptFunRecord("102", "twiss/y0=250E-6", mTD+4pt+blue, 0., 2.93679e-3, 5.59250e-4, -4.97855e-5, 5.58013e-6, -6.95230e-2, 1.30440e-3, -3.97470e+0, 6.53173e+0, -2.99717e+0, 1.63656e+1));
optFunRecords.push(OptFunRecord("103", "twiss/y0=250E-6", mTD+4pt+blue, 0., 2.50241e-3, 6.07936e-4, -4.97855e-5, 5.58013e-6, -6.57889e-2, 1.34936e-3, -4.05770e+0, 4.47299e+0, -3.22849e+0, 1.47176e+1));


// y0 = 550 um
optFunRecords.push(OptFunRecord("102", "twiss/y0=550E-6", mCi+3pt+red, 0., 2.94446e-3, -3.39902e-4, -4.97714e-5, -2.37342e-6, -6.95286e-2, 4.23439e-3, -3.97470e+0, 6.53173e+0, -2.99717e+0, 1.63656e+1));
optFunRecords.push(OptFunRecord("103", "twiss/y0=550E-6", mCi+3pt+red, 0., 2.51020e-3, -3.60610e-4, -4.97714e-5, -2.37342e-6, -6.57942e-2, 4.40954e-3, -4.05770e+0, 4.47299e+0, -3.22849e+0, 1.47176e+1));

optFunRecords.push(OptFunRecord("102", "twiss/y0=550E-6", mCi+3pt+red, 0.05, 6.59385e-3, -5.78776e-4, -7.07418e-5, -3.46342e-6, -7.22745e-2, 4.75881e-3, -4.18582e+0, -9.44711e+0, -3.52318e+0, -3.24459e+0));
optFunRecords.push(OptFunRecord("103", "twiss/y0=550E-6", mCi+3pt+red, 0.05, 5.97662e-3, -6.08994e-4, -7.07418e-5, -3.46342e-6, -6.80396e-2, 4.93555e-3, -4.25765e+0, -1.16936e+1, -3.77651e+0, -5.95435e+0));

optFunRecords.push(OptFunRecord("102", "twiss/y0=550E-6", mCi+3pt+red, 0.1, 1.07873e-2, -8.60801e-4, -9.61002e-5, -4.59107e-6, -7.59309e-2, 5.34710e-3, -4.38183e+0, -2.71793e+1, -4.13329e+0, -2.57216e+1));
optFunRecords.push(OptFunRecord("103", "twiss/y0=550E-6", mCi+3pt+red, 0.1, 9.94881e-3, -9.00858e-4, -9.61002e-5, -4.59107e-6, -7.10864e-2, 5.51500e-3, -4.43902e+0, -2.95252e+1, -4.40846e+0, -2.95449e+1));

// ----- beam 2 -----

// y0 = 0 ??

optFunRecords.push(OptFunRecord("2", "twiss/y0=0", mTU+4pt+black, 0., 3.51360e-3, -6.41999e-5, 3.22245e-5, -1.13594e-5, 9.39196e-2, -2.84189e-4, -3.94574e+0, 6.25620e+0, -3.07477e+0, 1.77672e+1));
optFunRecords.push(OptFunRecord("3", "twiss/y0=0", mTU+4pt+black, 0., 3.23244e-3, -1.63310e-4, 3.22245e-5, -1.13594e-5, 9.27120e-2, -1.70147e-4, -4.11385e+0, 4.31151e+0, -3.27072e+0, 1.60618e+1));

optFunRecords.push(OptFunRecord("2", "twiss/y0=0", mTU+4pt+black, 0.05, 8.53745e-3, -4.04063e-5, 3.73544e-5, -1.20230e-5, 1.01829e-1, -3.32551e-4, -4.15490e+0, -9.58285e+0, -3.62689e+0, -2.42237e+0));
optFunRecords.push(OptFunRecord("3", "twiss/y0=0", mTU+4pt+black, 0.05, 8.21153e-3, -1.45307e-4, 3.73544e-5, -1.20230e-5, 1.00491e-1, -2.07193e-4, -4.32188e+0, -1.20679e+1, -3.83869e+0, -4.96946e+0));

//----------------------------------------------------------------------------------------------------

for (int rpi : rp_tags.keys)
{
	NewRow();

	NewPad(false);
	label("{\SetFontSizesXX " + rp_labels[rpi] + "}");

	//--------------------

	NewPad("$\xi$", "$x_0\ung{mm}$");

	for (int fi : files.keys)
		draw(scale(1., +1e3), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_x0_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.x * 1e3), r.m);
		}
	}

	//--------------------

	NewPad("$\xi$", "$y_0\ung{mm}$");

	for (int fi : files.keys)
		draw(scale(1., 1e3), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_y0_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.y * 1e3), r.m);
		}
	}

	//--------------------

	NewPad("$\xi$", "$v_x$");

	for (int fi : files.keys)
		draw(scale(1., 1.), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_v_x_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.vx), r.m);
		}
	}

	//--------------------

	NewPad("$\xi$", "$v_y$");

	for (int fi : files.keys)
		draw(scale(1., 1.), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_v_y_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.vy), r.m);
		}
	}

	//--------------------

	NewPad("$\xi$", "$L_x\ung{mm}$");

	for (int fi : files.keys)
		draw(scale(1., 1.), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_L_x_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.Lx), r.m);
		}
	}

	//--------------------

	NewPad("$\xi$", "$L_y\ung{mm}$");

	for (int fi : files.keys)
		draw(scale(1., 1.), RootGetObject(files[fi], "RP"+rp_tags[rpi]+"/g_L_y_vs_xi"), f_pens[fi]);

	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi])
		{
			draw((r.xi, r.Ly), r.m);
		}
	}

	//--------------------

	NewPad(false);

	AddToLegend("<optics parametrisation");
	for (int fi : files.keys)
		AddToLegend(f_labels[fi], f_pens[fi]);	

	AddToLegend("<MAD-X/twiss");
	for (OptFunRecord r : optFunRecords)
	{
		if (r.rpId == rp_tags[rpi] && r.xi == 0)
		{
			AddToLegend(r.desc, r.m);
		}
	}

	AttachLegend();
}
