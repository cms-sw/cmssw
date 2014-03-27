#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

void LHEEventProduct::const_iterator::next()
{
	const lhef::HEPEUP &hepeup = event->hepeup();
	int line = this->line++;

	if (!line) {
		std::ostringstream ss;
		ss << std::setprecision(7)
		   << std::scientific
		   << std::uppercase
		   << "    " << hepeup.NUP
		   << "  " << hepeup.IDPRUP
		   << "  " << event->originalXWGTUP()
		   << "  " << hepeup.SCALUP
		   << "  " << hepeup.AQEDUP
		   << "  " << hepeup.AQCDUP << std::endl;
		tmp = ss.str();
		return;
	}
	line--;

	if (line < hepeup.NUP) {
		std::ostringstream ss;
		ss << std::setprecision(10)
		   << std::scientific
		   << std::uppercase
		   << "\t" << hepeup.IDUP[line]
		   << "\t" << hepeup.ISTUP[line]
		   << "\t" << hepeup.MOTHUP[line].first
		   << "\t" << hepeup.MOTHUP[line].second
		   << "\t" << hepeup.ICOLUP[line].first
		   << "\t" << hepeup.ICOLUP[line].second
		   << "\t" << hepeup.PUP[line][0]
		   << "\t" << hepeup.PUP[line][1]
		   << "\t" << hepeup.PUP[line][2]
		   << "\t" << hepeup.PUP[line][3]
		   << "\t" << hepeup.PUP[line][4]
		   << std::setprecision(3)
		   << "\t" << hepeup.VTIMUP[line]
		   << std::setprecision(1)
		   << std::fixed
		   << "\t" << hepeup.SPINUP[line] << std::endl;
		tmp = ss.str();
		return;
	}
	line -= hepeup.NUP;

	if (event->pdf()) {
		if (!line) {
			const PDF &pdf = *event->pdf();
			std::ostringstream ss;
			ss << std::setprecision(7)
			   << std::scientific
			   << std::uppercase
			   << "#pdf  " << pdf.id.first
			   << "  " << pdf.id.second
			   << "  " << pdf.x.first
			   << "  " << pdf.x.second
			   << "  " << pdf.scalePDF
			   << "  " << pdf.xPDF.first
			   << "  " << pdf.xPDF.second << std::endl;
			tmp = ss.str();
			return;
		}
		line--;
	}

	if (line < (int)event->comments_size()) {
		tmp = *(event->comments_begin() + line);
		return;
	}
	line -= event->comments_size();

	if (!line) {
		tmp = "</event>\n";
		return;
	}

	tmp.clear();
	this->line = npos;
}

LHEEventProduct::const_iterator LHEEventProduct::begin() const
{
	const_iterator result;
	result.event = this;
	result.line = 0;
	result.tmp = "<event>\n";
	return result;
}
