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

static std::vector<std::string> checklist{"iseed","Random",".log",".dat",".lhe"};
static std::vector<std::string> tag_comparison_checklist{"","MGRunCard","mgruncard"};

bool LHERunInfoProduct::find_if_checklist(const std::string x, std::vector<std::string> checklist) {
    return checklist.end() != std::find_if(checklist.begin(),checklist.end(),[&](const std::string& y)
                                            { return x.find(y) != std::string::npos; } );
}

bool LHERunInfoProduct::isTagComparedInMerge(const std::string& tag) {
        return !(tag.empty() || tag.find("Alpgen") == 0 || tag == "MGGridCard" || tag=="MGRunCard" || tag == "mgruncard" || tag=="MadSpin" || tag=="madspin");
}

bool LHERunInfoProduct::mergeProduct(const LHERunInfoProduct &other)
{

  if (heprup_.IDBMUP != other.heprup_.IDBMUP ||
	    heprup_.EBMUP != other.heprup_.EBMUP ||
	    heprup_.PDFGUP != other.heprup_.PDFGUP ||
	    heprup_.PDFSUP != other.heprup_.PDFSUP ||
	    heprup_.IDWTUP != other.heprup_.IDWTUP) {
        
	  return false;	
	}

	bool compatibleHeaders = (headers_ == other.headers_);

	// try to merge not equal but compatible headers (i.e. different iseed)
	while(!compatibleHeaders) {
		// okay, something is not the same.
		// Let's try to merge, but don't duplicate identical headers
		// and test the rest against a whitelist

		std::set<Header, HeaderLess> headers;
		std::copy(headers_begin(), headers_end(),
		          std::inserter(headers, headers.begin()));

    // make a list of headers contained in the second file
    std::vector<std::vector<std::string> > runcard_v2;
    std::vector<std::string> runcard_v2_header;
    for(auto header2 : headers_) {
        // fill a vector with the relevant header tags that can be not equal but sill compatible
        if(find_if_checklist(header2.tag(),tag_comparison_checklist)){
          runcard_v2.push_back(header2.lines());
          runcard_v2_header.push_back(header2.tag());
        }
    }
    
    // loop over the headers of the original file
		bool failed = false;
		for(std::vector<LHERunInfoProduct::Header>::const_iterator
					header = other.headers_begin();
		    header != other.headers_end(); ++header) {
          
			if (headers.count(*header)) {
				continue;
			}
            
      if(find_if_checklist(header->tag(),tag_comparison_checklist)){
        bool header_compatible = false;
        for (unsigned int iter_runcard = 0; iter_runcard < runcard_v2.size(); iter_runcard++){
          
          std::vector<std::string> runcard_v1 = header->lines();
          runcard_v1.erase( std::remove_if( runcard_v1.begin(), runcard_v1.end(), 
                                            [&](const std::string& x){ return find_if_checklist(x,checklist); } ), 
                                            runcard_v1.end() );
          runcard_v2[iter_runcard].erase( std::remove_if( runcard_v2[iter_runcard].begin(), runcard_v2[iter_runcard].end(), 
                                            [&](const std::string& x){ return find_if_checklist(x,checklist); } ), 
                                            runcard_v2[iter_runcard].end() );
          
          if(std::equal(runcard_v1.begin(), runcard_v1.end(), runcard_v2[iter_runcard].begin())){
            header_compatible = true;
            break;
          }
        }
        if(header_compatible) continue;
      }
      
			if(isTagComparedInMerge(header->tag())){ 
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
    return false;
	}

	// it is exactly the same, so merge
	return true;
}

void LHERunInfoProduct::swap(LHERunInfoProduct& other) {
  heprup_.swap(other.heprup_);
  headers_.swap(other.headers_);
  comments_.swap(other.comments_);
}
