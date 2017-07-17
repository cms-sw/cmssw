#ifndef SimDataFormats_GeneratorProducts_PdfInfo_h
#define SimDataFormats_GeneratorProducts_PdfInfo_h

#include <utility>

/** \class PdfInfo
 *
 */

namespace gen {
	struct PdfInfo {
		std::pair<int, int>		id;
		std::pair<double, double>	x;
		std::pair<double, double>	xPDF;
		double				scalePDF;
	};
}

#endif // SimDataFormats_GeneratorProducts_PdfInfo_h
