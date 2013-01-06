#ifndef CategoryCriteria_h
#define CategoryCriteria_h

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Common/interface/Ref.h"


//! Implement a selector given a track or vertex collection and track or vertex classifier.
template <typename Collection, typename Classifier>
class CategoryCriteria
{

public:

    // Input collection type
    typedef Collection collection;

    // Type of the collection elements
    typedef typename Collection::value_type type;

    // Oumemberut collection type
    typedef std::vector<const type *> container;

    // Iterator over result collection type.
    typedef typename container::const_iterator const_iterator;

    // Constructor from parameter set configurability
    CategoryCriteria(const edm::ParameterSet & config) :
            classifier_(config),
            evaluate_( config.getParameter<std::string>("cut") ) {}

    // Select object from a collection and possibly event content
    void select(const edm::Handle<collection> & collectionHandler, const edm::Event & event, const edm::EventSetup & setup)
    {

        selected_.clear();

        // const collection & collectionPointer = *(collectionHandler.product());

        classifier_.newEvent(event, setup);

        for (typename collection::size_type i = 0; i < collectionHandler->size(); ++i)
        {
            edm::Ref<Collection> member(collectionHandler, i);

            classifier_.evaluate(member);

            // Classifier is evaluated using StringCutObjectSelector
            if ( evaluate_(classifier_) )
                selected_.push_back( &(*member) );
        }
    }

    // Iterators over selected objects: collection begin
    const_iterator begin() const
    {
        return selected_.begin();
    }

    // Iterators over selected objects: collection end
    const_iterator end() const
    {
        return selected_.end();
    }

    // True if no object has been selected
    std::size_t size() const
    {
        return selected_.size();
    }

private:

    container selected_;

    Classifier classifier_;

    StringCutObjectSelector<typename Classifier::Categories> evaluate_;

};


#endif
