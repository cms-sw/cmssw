#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

bool LHERunInfoProduct::const_iterator::operator ==
					(const const_iterator &other) const
{
	if (mode != other.mode)
		return false;

	switch(mode) {
	    case kFooter:
	    case kDone:
		return true;

	    case kHeader:
		return header == other.header;

	    case kBody:
		return header == other.header && iter == other.iter;

	    case kInit:
		return line == other.line;
	}

	return false;
}

void LHERunInfoProduct::const_iterator::next()
{
	tmp.clear();

	do {
		switch(mode) {
		    case kHeader:
			if (header == runInfo->headers_end()) {
				mode = kInit;
				tmp = "<init>\n";
				line = 0;
				break;
			} else {
				mode = kBody;
				const std::string &tag = header->tag();
				tmp = tag.empty() ? "<!--" : ("<" + tag + ">");
				iter = header->begin();
				continue;
			}

		    case kBody:
			if (iter == header->end()) {
				mode = kHeader;
				const std::string &tag = header->tag();
				tmp += tag.empty() ? "-->" : ("</" + tag + ">");
				tmp += "\n";
				header++;
			} else {
				tmp += *iter++;
				if (iter == header->end() &&
				    (tmp.empty() ||
				     (tmp[tmp.length() - 1] != '\r' &&
				      tmp[tmp.length() - 1] != '\n')))
					continue;
			}
			break;

		    case kInit: {
			const lhef::HEPRUP &heprup = runInfo->heprup();
			if (!line++) {
				std::ostringstream ss;
				ss << std::setprecision(7)
				   << std::scientific
				   << std::uppercase
				   << "    " << heprup.IDBMUP.first
				   << "  " << heprup.IDBMUP.second
				   << "  " << heprup.EBMUP.first
				   << "  " << heprup.EBMUP.second
				   << "  " << heprup.PDFGUP.first
				   << "  " << heprup.PDFGUP.second
				   << "  " << heprup.PDFSUP.first
				   << "  " << heprup.PDFSUP.second
				   << "  " << heprup.IDWTUP
				   << "  " << heprup.NPRUP << std::endl;
				tmp = ss.str();
				break;
			}
			if (line >= (unsigned int)heprup.NPRUP +
			            runInfo->comments_size() + 2) {
				tmp = "</init>\n";
				mode = kFooter;
				break;
			} else if (line >= (unsigned int)heprup.NPRUP + 2) {
				tmp = *(runInfo->comments_begin() + (line -
				             (unsigned int)heprup.NPRUP - 2));
				break;
			}

			std::ostringstream ss;
			ss << std::setprecision(7)
			   << std::scientific
			   << std::uppercase
			   << "\t" << heprup.XSECUP[line - 2]
			   << "\t" << heprup.XERRUP[line - 2]
			   << "\t" << heprup.XMAXUP[line - 2]
			   << "\t" << heprup.LPRUP[line - 2] << std::endl;
			tmp = ss.str();
		    }	break;

		    case kFooter:
			mode = kDone;

		    default:
			/* ... */;
		}
	} while(false);
}

LHERunInfoProduct::const_iterator LHERunInfoProduct::begin() const
{
	const_iterator result;

	result.runInfo = this;
	result.header = headers_begin();
	result.mode = const_iterator::kHeader;
	result.line = 0;
	result.tmp = "<LesHouchesEvents version=\"1.0\">\n";

	return result;
}

LHERunInfoProduct::const_iterator LHERunInfoProduct::init() const
{
	const_iterator result;

	result.runInfo = this;
	result.mode = const_iterator::kInit;
	result.line = 0;
	result.tmp = "<init>\n";

	return result;
}

const std::string &LHERunInfoProduct::endOfFile()
{
	static const std::string theEnd("</LesHouchesEvents>\n");

	return theEnd;
}
