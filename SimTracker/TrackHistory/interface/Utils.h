#ifndef Utils_h
#define Utils_h

//! Generic matching function
template<typename Reference, typename Association>
  std::pair<typename Association::data_type::first_type, double> match (Reference key, Association association, bool bestMatchByMaxValue)
{
    typename Association::data_type::first_type value;

    typename Association::const_iterator pos = association.find(key);

    if (pos == association.end()) return std::pair<typename Association::data_type::first_type, double> (value, 0);

    const std::vector<typename Association::data_type> & matches = pos->val;

    double q = bestMatchByMaxValue ? -1e30 : 1e30;

    for (std::size_t i = 0; i < matches.size(); ++i)
        if (bestMatchByMaxValue ? (matches[i].second > q) : (matches[i].second < q))
        {
            value = matches[i].first;
            q = matches[i].second;
        }

    return std::pair<typename Association::data_type::first_type, double> (value, q);
}

#endif
