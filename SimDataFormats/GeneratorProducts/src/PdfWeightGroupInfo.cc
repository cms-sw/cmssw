#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"

namespace gen {
  PdfWeightGroupInfo* PdfWeightGroupInfo::clone() const { return new PdfWeightGroupInfo(*this); }
  void PdfWeightGroupInfo::addLhaid(int lhaid) {
    lhaids_.push_back(lhaid);
    if (lhaids_.size() == parentLhapdfSize_)
      setIsWellFormed(true);
    else
      setIsWellFormed(false);
  }

  void PdfWeightGroupInfo::setParentLhapdfInfo(int lhaid) {
    parentLhapdfId_ = lhaid;
    LHAPDF::PDFSet pdfSet(LHAPDF::lookupPDF(lhaid).first);
    parentLhapdfSize_ = pdfSet.size();
    parentLhapdfError_ = pdfSet.errorType();
  }

}  // namespace gen
