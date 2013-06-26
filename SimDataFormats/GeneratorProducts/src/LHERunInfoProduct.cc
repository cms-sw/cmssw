#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <map>
#include <set>

#include "FWCore/Utilities/interface/Exception.h"

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
				if (line++ == 1)
					tmp = "</header>\n";
				else {
					mode = kInit;
					tmp = "<init>\n";
					line = 0;
				}
				break;
			} else if (!line) {
				line++;
				tmp = "<header>\n";
				break;
			} else {
				mode = kBody;
				const std::string &tag = header->tag();
				tmp = tag.empty() ? "<!--" :
				      (tag == "<>") ? "" : ("<" + tag + ">");
				iter = header->begin();
				continue;
			}

		    case kBody:
			if (iter == header->end()) {
				mode = kHeader;
				const std::string &tag = header->tag();
				tmp += tag.empty() ? "-->" :
				       (tag == "<>") ? "" : ("</" + tag + ">");
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

namespace {
	struct XSec {
		inline XSec() : xsec(0.0), err(0.0), max(0.0) {}

		double	xsec;
		double	err;
		double	max;
	};

	struct HeaderLess {
		bool operator() (const LHERunInfoProduct::Header &a,
		                 const LHERunInfoProduct::Header &b) const;
	};
}

bool HeaderLess::operator() (const LHERunInfoProduct::Header &a,
                             const LHERunInfoProduct::Header &b) const
{
	if (a == b)
		return false;
	if (a.tag() < b.tag())
		return true;
	if (a.tag() > b.tag())
		return false;

	LHERunInfoProduct::Header::const_iterator iter1 = a.begin();
	LHERunInfoProduct::Header::const_iterator iter2 = b.begin();

	for(; iter1 != a.end() && iter2 != b.end(); ++iter1, ++iter2) {
		if (*iter1 < *iter2)
			return true;
		else if (*iter1 != *iter2)
			return false;
	}

	return iter2 != b.end();
}

bool LHERunInfoProduct::isTagComparedInMerge(const std::string& tag) {
	return !(tag == "" || tag.find("Alpgen") == 0 || tag == "MGGridCard" || tag == "MGGenerationInfo");
}

bool LHERunInfoProduct::mergeProduct(const LHERunInfoProduct &other)
{
	if (heprup_.IDBMUP != other.heprup_.IDBMUP ||
	    heprup_.EBMUP != other.heprup_.EBMUP ||
	    heprup_.PDFGUP != other.heprup_.PDFGUP ||
	    heprup_.PDFSUP != other.heprup_.PDFSUP ||
	    heprup_.IDWTUP != other.heprup_.IDWTUP) {
		throw cms::Exception("ProductsNotMergeable")
			<< "Error in LHERunInfoProduct: LHE headers differ. "
			   "Cannot merge products." << std::endl;
	}

	bool compatibleHeaders = headers_ == other.headers_;

	// try to merge different, but compatible headers
	while(!compatibleHeaders) {
		// okay, something is different.
		// Let's try to merge, but don't duplicate identical headers
		// and test the rest against a whitelist

		std::set<Header, HeaderLess> headers;
		std::copy(headers_begin(), headers_end(),
		          std::inserter(headers, headers.begin()));

		bool failed = false;
		for(std::vector<LHERunInfoProduct::Header>::const_iterator
					header = other.headers_begin();
		    header != other.headers_end(); ++header) {
			if (headers.count(*header)) {
				continue;
			}

			if(isTagComparedInMerge(header->tag())) {
				failed = true;
			} else {
				addHeader(*header);	
				headers.insert(*header);
			}
		}
		if (failed) {
			break;
		}

		compatibleHeaders = true;
	}

	// still not compatible after fixups
	if (!compatibleHeaders) {
		throw cms::Exception("ProductsNotMergeable")
			<< "Error in LHERunInfoProduct: LHE headers differ. "
			   "Cannot merge products." << std::endl;
	}

	// it is exactly the same, so merge
	if (heprup_ == other.heprup_)
		return true;

	// the input files are different ones, presumably generation
	// of the same process in different runs with identical run number
	// attempt merge of processes and cross-sections

	std::map<int, XSec> processes;

	for(int i = 0; i < heprup_.NPRUP; i++) {
		int id = heprup_.LPRUP[i];
		XSec &x = processes[id];
		x.xsec = heprup_.XSECUP[i];
		x.err = heprup_.XERRUP[i];
		x.max = heprup_.XMAXUP[i];
	}

	for(int i = 0; i < other.heprup_.NPRUP; i++) {
		int id = other.heprup_.LPRUP[i];
		XSec &x = processes[id];
		if (x.xsec) {
			double wgt1 = 1.0 / (x.err * x.err);
			double wgt2 = 1.0 / (other.heprup_.XERRUP[i] *
			                     other.heprup_.XERRUP[i]);
			x.xsec = (wgt1 * x.xsec +
			          wgt2 * other.heprup_.XSECUP[i]) /
			         (wgt1 + wgt2);
			x.err = 1.0 / std::sqrt(wgt1 + wgt2);
			x.max = std::max(x.max, other.heprup_.XMAXUP[i]);
		} else {
			x.xsec = other.heprup_.XSECUP[i];
			x.err = other.heprup_.XERRUP[i];
			x.max = other.heprup_.XMAXUP[i];
		}
	}

	heprup_.resize(processes.size());
	unsigned int i = 0;
	for(std::map<int, XSec>::const_iterator iter = processes.begin();
	    iter != processes.end(); ++iter, i++) {
		heprup_.LPRUP[i] = iter->first;
		heprup_.XSECUP[i] = iter->second.xsec;
		heprup_.XERRUP[i] = iter->second.err;
		heprup_.XMAXUP[i] = iter->second.max;
	}

	return true;
}
