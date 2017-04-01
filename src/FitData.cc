/*********************************************************
 * $Id: RPXMLConfig.h,v 1.3 2006/08/17 15:54:21 lgrzanka Exp $
 * $Revision: 1.3 $
 * $Date: 2006/08/17 15:54:21 $
 **********************************************************/

#include "FitData.h"

FitData::FitData() {
}
;

FitData::~FitData() {
}
;

void FitData::readIn(std::string name) {

	int n = 0, k = 0;

	std::ifstream ifs;
	ifs.open(name.c_str());

	long int i;

	std::string form1 = "%*s %le %le %le %le %*le %le";
	std::string ln1;

	Double_t x1, px1, y1, py1, pt1;

	getline(ifs, ln1);
	k++;

	// skip comment line
	while (ln1[0] == '@') {
		getline(ifs, ln1);
		k++;
	}

	// skip 2 lines
	if (ln1[0] == '*') {
		getline(ifs, ln1);
		k++;
	}

	if (ln1[0] == '$') {
		getline(ifs, ln1);
		k++;
	}

	while (!ifs.eof()) {
		getline(ifs, ln1);
		n++;
	}

	this->inSize = n;
	std::cout << "size, n=" << n << std::endl;

	ifs.close();
	std::ifstream ifs1;
	ifs1.open(name.c_str());

	this->dataIn.ResizeTo(6, n);

	for (i = 0; i < k - 1; i++) {
		getline(ifs1, ln1);
	}

	for (i = 0; i < this->dataIn.GetNcols(); i++) {
		getline(ifs1, ln1);
		sscanf(ln1.c_str(), form1.c_str(), &x1, &px1, &y1, &py1, &pt1);
		(this->dataIn)[0][i] = 0;
		(this->dataIn)[1][i] = x1;
		(this->dataIn)[2][i] = px1;
		(this->dataIn)[3][i] = y1;
		(this->dataIn)[4][i] = py1;
		(this->dataIn)[5][i] = pt1;
	}
	std::cout << "Data read in." << std::endl;
	ifs1.close();
}
;

int FitData::readOut(std::string name, std::string section) {
	std::cout << "readOut::start\n";
	int m = 0, n = 0, k = 0;
	std::ifstream ifs;
	ifs.open(name.c_str());

	long int i;

	std::string form1 = "%hd %*hd %le %le %le %le %*le %le %*le";
	std::string form2 = "%*s %*hd %*hd %hd %hd %s";
	std::string ln1;
	char ln2[200];
	char ln3[200];

	Double_t x1, px1, y1, py1, pt1;

	getline(ifs, ln1);
	k++;

	// skip comment line
	while (ln1[0] == '@') {
		getline(ifs, ln1);
		k++;
	}

	// skip 2 lines
	if (ln1[0] == '*') {
		getline(ifs, ln1);
		k++;
	}

	if (ln1[0] == '$') {
		getline(ifs, ln1);
		k++;
	}
	Int_t notFound = 1;
	std::cout << "section: " << section << "\n";
	while (ifs.good() && !ifs.eof() && notFound) {
		if (ln1[0] == '#') {
			sscanf(ln1.c_str(), "%*s %*hd %*hd %hd %hd %s", &m, &n, &ln2);
			if (boost::iequals(ln2, section)) {
				this->outSize = m;
				std::cout << m << " particles outcoming" << std::endl;
				notFound = 0;
			}
		}
		getline(ifs, ln1);
		n++;
	}

	if (notFound) {
		std::cout << "Section " << section << " not found" << std::endl;
	}

	this->dataOut.ResizeTo(6, m);
	for (i = 0; i < this->dataOut.GetNcols(); i++) {
		sscanf(ln1.c_str(), form1.c_str(), &n, &x1, &px1, &y1, &py1, &pt1);
		(this->dataOut)[0][i] = n;
		(this->dataOut)[1][i] = x1;
		(this->dataOut)[2][i] = px1;
		(this->dataOut)[3][i] = y1;
		(this->dataOut)[4][i] = py1;
		(this->dataOut)[5][i] = pt1;
		(this->dataIn)[0][n - 1] = n;
		getline(ifs, ln1);
	}
	ifs.close();
	std::cout << "readOut::end\n";
}
;

void FitData::readAdditionalScoringPlanes(std::string name,
		const std::vector<std::string> &sections) {
	std::cout << "readAdditionalScoringPlanes:start\n";
	int m = 0, n = 0, k = 0;

	std::ifstream ifs;
	ifs.open(name.c_str());

	long int i;

	std::string form1 = "%hd %*hd %le %le %le %le %*le %le %*le";
	std::string form2 = "%*s %*hd %*hd %hd %hd %s";
	std::string ln1;
	char ln2[200];
	char ln3[200];

	Double_t x1, px1, y1, py1, pt1;

	getline(ifs, ln1);
	k++;

	// skip comment line
	while (ln1[0] == '@') {
		getline(ifs, ln1);
		k++;
	}

	// skip 2 lines
	if (ln1[0] == '*') {
		getline(ifs, ln1);
		k++;
	}

	if (ln1[0] == '$') {
		getline(ifs, ln1);
		k++;
	}
	bool stop = false;
	while (ifs.good() && !ifs.eof()) {
		stop = false;
		if (ln1[0] == '#') {
			sscanf(ln1.c_str(), "%*s %*hd %*hd %hd %hd %s", &m, &n, &ln2);
			for (int i = 0; i < sections.size() && !stop; i++) {
				if (boost::iequals(ln2, sections[i])) {
					this->outSize = m;
					TMatrixD* curr_data_set =
							&additional_scoring_planes[sections[i]];
					curr_data_set->ResizeTo(6, m);
					for (i = 0; i < (*curr_data_set).GetNcols(); i++) {
						getline(ifs, ln1);
						sscanf(ln1.c_str(), form1.c_str(), &n, &x1, &px1, &y1,
								&py1, &pt1);
						(*curr_data_set)[0][i] = n;
						(*curr_data_set)[1][i] = x1;
						(*curr_data_set)[2][i] = px1;
						(*curr_data_set)[3][i] = y1;
						(*curr_data_set)[4][i] = py1;
						(*curr_data_set)[5][i] = pt1;
					}
					stop = true;
				}
			}
		}
		getline(ifs, ln1);
		n++;
	}
	ifs.close();
	std::cout << "readAdditionalScoringPlanes:end\n";
}

void FitData::readLost(std::string name) {
	int n = 0, k = 0;

	std::ifstream ifs;
	ifs.open(name.c_str());

	long int i;

	std::string form1 = "%d %*d %le  %le  %le  %le  %*le  %le  %le  %*le \"%s";
	std::string ln1;

	getline(ifs, ln1);
	k++;

	// skip comment line
	while (ln1[0] == '@') {
		getline(ifs, ln1);
		k++;
	}

	// skip 2 lines
	if (ln1[0] == '*') {
		getline(ifs, ln1);
		k++;
	}

	if (ln1[0] == '$') {
		getline(ifs, ln1);
		k++;
	}

	while (!ifs.eof()) {
		getline(ifs, ln1);
		n++;
	}

	this->lostSize = n;

	ifs.close();
	std::ifstream ifs1;
	ifs1.open(name.c_str());

	std::cout << n << " particles lost" << std::endl;

	for (i = 0; i < k - 1; i++) {
		getline(ifs1, ln1);
	}

	lost_particle part;
	char element[1024];
	for (i = 0; i < this->lostSize; i++) {
		getline(ifs1, ln1);
		sscanf(ln1.c_str(), form1.c_str(), &part.part_id, &part.x1, &part.px1,
				&part.y1, &part.py1, &part.pt1, &part.s, element);
		element[strlen(element) - 1] = 0;
		part.element = element;

		(this->dataLost).push_back(part);
	}
	ifs1.close();
}

void FitData::writeIn(std::string name) {
	long int i;
	std::ofstream ofs;
	std::string form2 = "%hd %le %le %le %le %le";
	char ln2[200];
	ofs.open(name.c_str(), std::ios_base::app);
	for (i = 0; i < this->dataIn.GetNcols(); i++) {
		if ((this->dataIn)[0][i] > 0) {
			sprintf(ln2, form2.c_str(), (Int_t)(this->dataIn)[0][i],
					(this->dataIn)[1][i], (this->dataIn)[2][i],
					(this->dataIn)[3][i], (this->dataIn)[4][i],
					(this->dataIn)[5][i]);
			ofs << ln2 << std::endl;
		}
	}
	ofs.close();
}
;

void FitData::writeOut(std::string name) {
	long int i;
	std::ofstream ofs;
	std::string form2 = "%hd %le %le %le %le %le";
	char ln2[200];
	ofs.open(name.c_str(), std::ios_base::app);
	for (i = 0; i < this->dataOut.GetNcols(); i++) {
		sprintf(ln2, form2.c_str(), (Int_t)(this->dataOut)[0][i],
				(this->dataOut)[1][i], (this->dataOut)[2][i],
				(this->dataOut)[3][i], (this->dataOut)[4][i],
				(this->dataOut)[5][i]);
		ofs << ln2 << std::endl;
	}
	ofs.close();
}
;

TVectorD FitData::getSampleIn(int n) {

}
;

Double_t FitData::getOutVar(VarName id, int nr) {

}
;

TVectorD FitData::getOutVector(VarName id) {

}
;

TVectorD FitData::getInVector(VarName id) {

}
;

Int_t FitData::getInSize() {
	return this->inSize;
}
;

Int_t FitData::getOutSize() {
	return this->outSize;
}
;

int FitData::AppendRootFile(TTree *inp_tree, std::string data_prefix) {
	if (inp_tree == NULL)
		return 0;

	double in_var[6];
	double out_var[7];

	std::map < std::string, inter_plane_data > inter_planes;

	typedef std::map<std::string, TMatrixD>::iterator map_iter;

	for (map_iter it = additional_scoring_planes.begin();
			it != additional_scoring_planes.end(); it++) {
		inter_plane_data rec;
		rec.data = &(it->second);
		rec.x_lab = it->first + "_x_out";
		rec.theta_x_lab = it->first + "_theta_x_out";
		rec.y_lab = it->first + "_y_out";
		rec.theta_y_lab = it->first + "_theta_y_out";
		rec.ksi_lab = it->first + "_ksi_out";
		rec.s_lab = it->first + "_s_out";
		rec.valid_lab = it->first + "_valid_out";
		rec.x = rec.theta_x = rec.y = rec.theta_y = rec.s = rec.valid = 0.0;
		rec.cur_index = 0;

		inter_planes[it->first] = rec;
	}

	//in- out-lables
	std::string x_in_lab = "x_in";
	std::string theta_x_in_lab = "theta_x_in";
	std::string y_in_lab = "y_in";
	std::string theta_y_in_lab = "theta_y_in";
	std::string ksi_in_lab = "ksi_in";
	std::string s_in_lab = "s_in";

	std::string x_out_lab = data_prefix + "_x_out";
	std::string theta_x_out_lab = data_prefix + "_theta_x_out";
	std::string y_out_lab = data_prefix + "_y_out";
	std::string theta_y_out_lab = data_prefix + "_theta_y_out";
	std::string ksi_out_lab = data_prefix + "_ksi_out";
	std::string s_out_lab = data_prefix + "_s_out";
	std::string valid_out_lab = data_prefix + "_valid_out";

	//disable not needed branches to speed up the readin
	inp_tree->SetBranchStatus("*", 0); //disable all branches
	inp_tree->SetBranchStatus(x_in_lab.c_str(), 1);
	inp_tree->SetBranchStatus(theta_x_in_lab.c_str(), 1);
	inp_tree->SetBranchStatus(y_in_lab.c_str(), 1);
	inp_tree->SetBranchStatus(theta_y_in_lab.c_str(), 1);
	inp_tree->SetBranchStatus(ksi_in_lab.c_str(), 1);
	inp_tree->SetBranchStatus(x_out_lab.c_str(), 1);
	inp_tree->SetBranchStatus(theta_x_out_lab.c_str(), 1);
	inp_tree->SetBranchStatus(y_out_lab.c_str(), 1);
	inp_tree->SetBranchStatus(theta_y_out_lab.c_str(), 1);
	inp_tree->SetBranchStatus(ksi_out_lab.c_str(), 1);
	inp_tree->SetBranchStatus(valid_out_lab.c_str(), 1);

	//set input data adresses
	inp_tree->SetBranchAddress(x_in_lab.c_str(), &(in_var[0]));
	inp_tree->SetBranchAddress(theta_x_in_lab.c_str(), &(in_var[1]));
	inp_tree->SetBranchAddress(y_in_lab.c_str(), &(in_var[2]));
	inp_tree->SetBranchAddress(theta_y_in_lab.c_str(), &(in_var[3]));
	inp_tree->SetBranchAddress(ksi_in_lab.c_str(), &(in_var[4]));
	inp_tree->SetBranchAddress(s_in_lab.c_str(), &(in_var[5]));

	//set output data adresses
	inp_tree->SetBranchAddress(x_out_lab.c_str(), &(out_var[0]));
	inp_tree->SetBranchAddress(theta_x_out_lab.c_str(), &(out_var[1]));
	inp_tree->SetBranchAddress(y_out_lab.c_str(), &(out_var[2]));
	inp_tree->SetBranchAddress(theta_y_out_lab.c_str(), &(out_var[3]));
	inp_tree->SetBranchAddress(ksi_out_lab.c_str(), &(out_var[4]));
	inp_tree->SetBranchAddress(s_out_lab.c_str(), &(out_var[5]));
	inp_tree->SetBranchAddress(valid_out_lab.c_str(), &(out_var[6]));

	//set interplane branches
	typedef std::map<std::string, inter_plane_data>::iterator inter_planes_iterator;
	for (inter_planes_iterator it = inter_planes.begin();
			it != inter_planes.end(); it++) {
		inp_tree->SetBranchStatus(it->second.x_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.theta_x_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.y_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.theta_y_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.ksi_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.s_lab.c_str(), 1);
		inp_tree->SetBranchStatus(it->second.valid_lab.c_str(), 1);

		inp_tree->SetBranchAddress(it->second.x_lab.c_str(), &(it->second.x));
		inp_tree->SetBranchAddress(it->second.theta_x_lab.c_str(),
				&(it->second.theta_x));
		inp_tree->SetBranchAddress(it->second.y_lab.c_str(), &(it->second.y));
		inp_tree->SetBranchAddress(it->second.theta_y_lab.c_str(),
				&(it->second.theta_y));
		inp_tree->SetBranchAddress(it->second.ksi_lab.c_str(),
				&(it->second.ksi));
		inp_tree->SetBranchAddress(it->second.s_lab.c_str(), &(it->second.s));
		inp_tree->SetBranchAddress(it->second.valid_lab.c_str(),
				&(it->second.valid));
	}

	Long64_t entries = inp_tree->GetEntries();
	for (int i = 0; i < this->dataOut.GetNcols(); i++) {
		out_var[0] = (this->dataOut)[1][i];
		out_var[1] = (this->dataOut)[2][i];
		out_var[2] = (this->dataOut)[3][i];
		out_var[3] = (this->dataOut)[4][i];
		out_var[4] = (this->dataOut)[5][i];
		out_var[5] = 0; //temporarily
		out_var[6] = 1.0;

		int in_index = (Int_t)(this->dataOut)[0][i] - 1;
		int in_part_number = (Int_t)(this->dataOut)[0][i];
		in_var[0] = (this->dataIn)[1][in_index];
		in_var[1] = (this->dataIn)[2][in_index];
		in_var[2] = (this->dataIn)[3][in_index];
		in_var[3] = (this->dataIn)[4][in_index];
		in_var[4] = (this->dataIn)[5][in_index];
		in_var[5] = 0; //temporarily

		//read interplane branches
		for (inter_planes_iterator it = inter_planes.begin();
				it != inter_planes.end(); it++) {
			while ((*it->second.data)[0][it->second.cur_index] != in_part_number) {
				it->second.cur_index++;
			}
			it->second.x = (*it->second.data)[1][it->second.cur_index];
			it->second.theta_x = (*it->second.data)[2][it->second.cur_index];
			it->second.y = (*it->second.data)[3][it->second.cur_index];
			it->second.theta_y = (*it->second.data)[4][it->second.cur_index];
			it->second.ksi = (*it->second.data)[5][it->second.cur_index];
			it->second.s = 0;
			it->second.valid = 1.0;

		}
		inp_tree->Fill();
	}
	return this->dataOut.GetNcols();
}

int FitData::AppendLostParticlesRootFile(TTree *lost_part_tree) {
	if (lost_part_tree == NULL)
		return 0;

	std::string br_x_in_lab = "in_x";
	std::string br_theta_x_in_lab = "in_theta_x";
	std::string br_y_in_lab = "in_y";
	std::string br_theta_y_in_lab = "in_theta_y";
	std::string br_ksi_in_lab = "in_ksi";
	std::string br_s_in_lab = "in_s";

	std::string br_x_out_lab = "out_x";
	std::string br_theta_x_out_lab = "out_theta_x";
	std::string br_y_out_lab = "out_y";
	std::string br_theta_y_out_lab = "out_theta_y";
	std::string br_ksi_out_lab = "out_ksi";
	std::string br_s_out_lab = "out_s";
	std::string br_element_out_lab = "out_element";

	double x_in, theta_x_in, y_in, theta_y_in, ksi_in, s_in;
	double x_out, theta_x_out, y_out, theta_y_out, ksi_out, s_out;
	char element_out[512];
	element_out[0] = 0;

	lost_part_tree->SetBranchAddress(br_x_in_lab.c_str(), &x_in);
	lost_part_tree->SetBranchAddress(br_theta_x_in_lab.c_str(), &theta_x_in);
	lost_part_tree->SetBranchAddress(br_y_in_lab.c_str(), &y_in);
	lost_part_tree->SetBranchAddress(br_theta_y_in_lab.c_str(), &theta_y_in);
	lost_part_tree->SetBranchAddress(br_ksi_in_lab.c_str(), &ksi_in);
	lost_part_tree->SetBranchAddress(br_s_in_lab.c_str(), &s_in);

	lost_part_tree->SetBranchAddress(br_x_out_lab.c_str(), &x_out);
	lost_part_tree->SetBranchAddress(br_theta_x_out_lab.c_str(), &theta_x_out);
	lost_part_tree->SetBranchAddress(br_y_out_lab.c_str(), &y_out);
	lost_part_tree->SetBranchAddress(br_theta_y_out_lab.c_str(), &theta_y_out);
	lost_part_tree->SetBranchAddress(br_ksi_out_lab.c_str(), &ksi_out);
	lost_part_tree->SetBranchAddress(br_s_out_lab.c_str(), &s_out);
	lost_part_tree->SetBranchAddress(br_element_out_lab.c_str(), element_out);
	for (int i = 0; i < dataLost.size(); i++) {
		int index = dataLost[i].part_id - 1;
		x_in = dataIn[1][index];
		theta_x_in = dataIn[2][index];
		y_in = dataIn[3][index];
		theta_y_in = dataIn[4][index];
		ksi_in = dataIn[5][index];
		s_in = 0;

		x_out = dataLost[i].x1;
		theta_x_out = dataLost[i].px1;
		y_out = dataLost[i].y1;
		theta_y_out = dataLost[i].py1;
		ksi_out = dataLost[i].pt1;
		s_out = dataLost[i].s;
		strncpy(element_out, dataLost[i].element.c_str(), 511);

		lost_part_tree->Fill();
	}
	return this->dataLost.size();
}

int FitData::AppendAcceleratorAcceptanceRootFile(TTree *acceptance_tree) {
	//x:theta_x:y:theta_y:ksi:mad_accept:par_accept
	double out[7];

	acceptance_tree->SetBranchAddress("x", &out[0]);
	acceptance_tree->SetBranchAddress("theta_x", &out[1]);
	acceptance_tree->SetBranchAddress("y", &out[2]);
	acceptance_tree->SetBranchAddress("theta_y", &out[3]);
	acceptance_tree->SetBranchAddress("ksi", &out[4]);
	acceptance_tree->SetBranchAddress("mad_accept", &out[5]);
	acceptance_tree->SetBranchAddress("par_accept", &out[6]);

	for (int i = 0; i < this->dataIn.GetNcols(); i++) {
		out[0] = (this->dataIn)[1][i];
		out[1] = (this->dataIn)[2][i];
		out[2] = (this->dataIn)[3][i];
		out[3] = (this->dataIn)[4][i];
		out[4] = (this->dataIn)[5][i];
		out[5] = ((this->dataIn)[0][i] > 0) ? (1.0) : (0);
		out[6] = 0.0;
		acceptance_tree->Fill();
	}
	return this->inSize;
}

