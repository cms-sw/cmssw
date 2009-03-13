#ifndef Utils_h
#define Utils_h

//! Generic matching function
template<typename Reference, typename Association>
typename Association::data_type::first_type match (Reference key, Association association, bool bestMatchByMaxValue)
{
    typename Association::data_type::first_type value;

    typename Association::const_iterator pos = association.find(key);

    if (pos == association.end()) return value;

    const std::vector<typename Association::data_type> & matches = pos->val;

    double m = bestMatchByMaxValue ? -1e30 : 1e30;

    for (std::size_t i = 0; i < matches.size(); ++i)
        if (bestMatchByMaxValue ? (matches[i].second > m) : (matches[i].second < m))
        {
            value = matches[i].first;
            m = matches[i].second;
        }

    return value;
}

#endif
