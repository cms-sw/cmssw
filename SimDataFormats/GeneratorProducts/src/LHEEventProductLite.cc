#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProductLite.h"

void LHEEventProductLite::const_iterator::next()
{
	const lhef::HEPEUP &hepeup = event_->hepeup();
	int line_ = this->line_++;

	if (!line_) {
		std::ostringstream ss;
		ss << std::setprecision(7)
		   << std::scientific
		   << std::uppercase
		   << "    " << hepeup.NUP
		   << "  " << hepeup.IDPRUP
		   << "  " << event_->originalXWGTUP()
		   << "  " << hepeup.SCALUP
		   << "  " << hepeup.AQEDUP
		   << "  " << hepeup.AQCDUP << std::endl;
		tmp_ = ss.str();
		return;
	}
	line_--;

	if (line_ < hepeup.NUP) {
		std::ostringstream ss;
		ss << std::setprecision(10)
		   << std::scientific
		   << std::uppercase
		   << "\t" << hepeup.IDUP[line_]
		   << "\t" << hepeup.ISTUP[line_]
		   << "\t" << hepeup.MOTHUP[line_].first
		   << "\t" << hepeup.MOTHUP[line_].second
		   << "\t" << hepeup.ICOLUP[line_].first
		   << "\t" << hepeup.ICOLUP[line_].second
		   << "\t" << hepeup.PUP[line_][0]
		   << "\t" << hepeup.PUP[line_][1]
		   << "\t" << hepeup.PUP[line_][2]
		   << "\t" << hepeup.PUP[line_][3]
		   << "\t" << hepeup.PUP[line_][4]
		   << std::setprecision(3)
		   << "\t" << hepeup.VTIMUP[line_]
		   << std::setprecision(1)
		   << std::fixed
		   << "\t" << hepeup.SPINUP[line_] << std::endl;
		tmp_ = ss.str();
		return;
	}
	line_ -= hepeup.NUP;

	if (event_->pdf()) {
		if (!line_) {
			const PDF &pdf = *event_->pdf();
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
			tmp_ = ss.str();
			return;
		}
		line_--;
	}

	if (!line_) {
		tmp_ = "</event>\n";
		return;
	}

	tmp_.clear();
	this->line_ = npos_;
}

LHEEventProductLite::const_iterator LHEEventProductLite::begin() const
{
	const_iterator result;
	result.event_ = this;
	result.line_ = 0;
	result.tmp_ = "<event>\n";
	return result;
}
