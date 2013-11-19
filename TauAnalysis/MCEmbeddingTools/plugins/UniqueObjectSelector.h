#ifndef TauAnalysis_MCEmbeddingTools_UniqueObjectSelector_h
#define TauAnalysis_MCEmbeddingTools_UniqueObjectSelector_h

/** \class UniqueObjectSelector
 *
 * Select unique object.
 * The objects contained in the input collection are first filtered by cuts
 * and then ranked according to user-defined criteria.
 * The highest ranked object passing all cuts is stored in the output collection.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: UniqueObjectSelector.h,v 1.1 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <vector>

template <typename T>
class UniqueObjectSelector : public edm::EDFilter
{
 public:
  explicit UniqueObjectSelector(const edm::ParameterSet&);
  ~UniqueObjectSelector();

 private:
  bool filter(edm::Event&, const edm::EventSetup&);

  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag src_;
  
  StringCutObjectSelector<T>* cut_;
  StringObjectFunction<T>* rank_;

  bool filter_;

  typedef std::vector<T> ObjectCollection;
};

#endif
