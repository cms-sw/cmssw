#ifndef TrackingTools_TrajectoryCleaning_src_OtherHashMaps
#define TrackingTools_TrajectoryCleaning_src_OtherHashMaps

#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace cmsutil {

  /*** The concept is the same of std::map<K, std::vector<V>>, but the implementation is much different (and the interface is not really compatible)
 *   It works only for K and V objects with trivial destructors (primitive types and bare pointers are fine)
 *   The main implementation difference w.r.t. unordered_map<K, std::vector<V>> is that this map is optimized to do very few memory allocations
 *   (in particular, clear does not redeme memory so if you just clear the map instead of deleting it you won't call malloc each time you fill it)
 *   The map doesn't rehash, so you should set a reasonable number of buckets (that you can change also when clearing the map)
 *   Note that too many buckets will make your map slow to clear (and possibly more prone to cache misses).
 *   Although it can take an allocator as template argument, it has been tested only with std::allocator.
 *   When used in the TrajectoryCleanerBySharedHits it works and it does improve the performance. Any other usecase was not checked at all.
 * */
  template <typename K,
            typename V,
            typename Hasher = std::hash<K>,
            typename Equals = std::equal_to<K>,
            typename Alloc = std::allocator<V> >
  class SimpleAllocHashMultiMap {
    // taking Alloc definition from http://www.codeproject.com/KB/cpp/allocator.aspx
    static_assert(std::conjunction<std::is_trivially_destructible<K>, std::is_trivially_destructible<V> >::value);

  public:
    typedef Hasher hasher;
    typedef SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc> map_type;
    typedef typename std::conditional<std::is_pointer<V>::value, V, V const &>::type
        value_ref;  // otherwise we get problems with 'const' if V is a pointer

    SimpleAllocHashMultiMap(size_t buckets, size_t keyRowSize, size_t valueRowSize, size_t maxRows = 50);

    ~SimpleAllocHashMultiMap();

    void clear(size_t newBucketSize = 0) {
      if (newBucketSize != 0) {
        if (newBucketSize > bucketCapacity_) {
          ptrAlloc_.deallocate(buckets_, bucketCapacity_);
          bucketCapacity_ = newBucketSize;
          buckets_ = ptrAlloc_.allocate(bucketCapacity_);
        }
        bucketSize_ = newBucketSize;
      }
      memset(buckets_, 0, bucketSize_ * sizeof(KeyItem *));
      currentKeyRow_ = keyRows_.begin();
      nextKeyItem_ = keyRows_.front();
      keyEndMarker_ = nextKeyItem_ + keyRowSize_;
      currentValueRow_ = valueRows_.begin();
      nextValueItem_ = valueRows_.front();
      valueEndMarker_ = nextValueItem_ + valueRowSize_;
      if ((keyRows_.size() > maxRows_) || (valueRows_.size() > maxRows_))
        freeRows();
    }
    void freeRows();

    bool empty() const { return nextKeyItem_ == keyRows_.front(); }

    void insert(K const &key, value_ref value);

    struct ValueItem {
      ValueItem(ValueItem *next1, value_ref val) : next(next1), value(val) {}
      ValueItem *next;
      V value;
      typedef V value_type;
      const value_type &operator()() const { return value; }
    };

    struct KeyItem {
      KeyItem(KeyItem *next1, K const &key1, ValueItem *val1) : key(key1), next(next1), value(val1) {}
      K key;
      KeyItem *next;
      ValueItem *value;
      typedef KeyItem value_type;
      const value_type &operator()() const { return *this; }
    };

    template <typename Item>
    class item_iterator {
    public:
      typedef ::std::forward_iterator_tag iterator_category;
      typedef const typename Item::value_type value_type;
      typedef const value_type &reference;
      typedef const value_type *pointer;
      typedef ptrdiff_t difference_type;
      typedef item_iterator<Item> self_type;

      item_iterator() : it_(nullptr) {}
      item_iterator(const Item *it) : it_(it) {}
      const value_type &operator*() const { return (*it_)(); }
      const value_type *operator->() const { return &(*it_)(); }
      self_type &operator++() {
        if (it_ != nullptr)
          it_ = it_->next;
        return *this;
      }
      bool operator==(const self_type &other) const { return it_ == other.it_; }
      bool operator!=(const self_type &other) const { return it_ != other.it_; }
      bool good() const { return (it_ != nullptr); }

    private:
      const Item *it_;
    };
    typedef item_iterator<ValueItem> value_iterator;

    value_iterator values(K const &key);

  private:
    using AllocTraits = std::allocator_traits<Alloc>;
    using KeyItemAllocator = typename AllocTraits::template rebind_alloc<KeyItem>;
    using KeyItemPtrAllocator = typename AllocTraits::template rebind_alloc<KeyItem *>;
    using ValueItemAllocator = typename AllocTraits::template rebind_alloc<ValueItem>;

    // --- buckets ---
    size_t bucketSize_, bucketCapacity_;
    KeyItem **buckets_;
    // --- keys ---
    size_t keyRowSize_;
    std::list<KeyItem *> keyRows_;
    typename std::list<KeyItem *>::iterator
        currentKeyRow_;  // last row that is currently in use. nextItem_ and the last valid item are both on this row. it is never rows_.end()
    KeyItem *nextKeyItem_, *keyEndMarker_;
    // --- values ---
    size_t valueRowSize_;
    std::list<ValueItem *> valueRows_;
    typename std::list<ValueItem *>::iterator
        currentValueRow_;  // last row that is currently in use. nextItem_ and the last valid item are both on this row. it is never rows_.end()
    ValueItem *nextValueItem_, *valueEndMarker_;
    // --- other ---
    size_t maxRows_;
    Hasher hasher_;
    Equals eq_;
    KeyItemAllocator keyAlloc_;
    ValueItemAllocator valueAlloc_;
    KeyItemPtrAllocator ptrAlloc_;

    KeyItem *push_back_(K const &key, KeyItem *next);
    ValueItem *push_back_(value_ref value, ValueItem *next);
    KeyItem &find_or_insert_(K const &key);

  public:
    void dump() {
      std::cout << "Dumping HASH MULTIMAP" << std::endl;
      std::cout << "  Key items: "
                << (std::distance(keyRows_.begin(), currentKeyRow_) * keyRowSize_ + (nextKeyItem_ - *currentKeyRow_))
                << std::endl;
      std::cout << "  Value items: "
                << (std::distance(valueRows_.begin(), currentValueRow_) * valueRowSize_ +
                    (nextValueItem_ - *currentValueRow_))
                << std::endl;
      size_t row = 0;
      std::cout << "  Buckets (size " << bucketSize_ << ", capacity " << bucketCapacity_ << ")";
      for (KeyItem **p = buckets_; p != buckets_ + bucketSize_; ++p, ++row) {
        std::cout << "      [" << row << "] " << *p << std::endl;
      }
      row = 0;
      std::cout << "  Key Items " << std::endl;
      for (typename std::list<KeyItem *>::iterator it = keyRows_.begin(), last = keyRows_.end(); it != last;
           ++it, ++row) {
        KeyItem *lastI = *it + keyRowSize_;
        std::cout << "   Key Row " << row << " (of size  " << keyRowSize_ << ")" << std::endl;
        for (KeyItem *p = *it; p != lastI; ++p) {
          std::cout << "      @ " << p << " [" << p->key << ", @" << p->value << "], next = " << p->next << std::endl;
          if ((it == currentKeyRow_) && (p == nextKeyItem_ - 1)) {
            std::cout << "      ^^^ this was the last valid item." << std::endl;
            last = 0;
            break;
          }
        }
        if (lastI == 0)
          break;
      }
      row = 0;
      std::cout << "  Value Items " << std::endl;
      for (typename std::list<ValueItem *>::iterator it = valueRows_.begin(), last = valueRows_.end(); it != last;
           ++it, ++row) {
        ValueItem *lastI = *it + valueRowSize_;
        std::cout << "   Value Row " << row << " (of size  " << valueRowSize_ << ")" << std::endl;
        for (ValueItem *p = *it; p != lastI; ++p) {
          std::cout << "      @ " << p << " [" << p->value << "], next = " << p->next << std::endl;
          if ((it == currentValueRow_) && (p == nextValueItem_ - 1)) {
            std::cout << "      ^^^ this was the last valid item." << std::endl;
            last = 0;
            break;
          }
        }
        if (lastI == 0)
          break;
      }
      std::cout << "  End of dump" << std::endl;
    }
  };

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::SimpleAllocHashMultiMap(size_t buckets,
                                                                                size_t keyRowSize,
                                                                                size_t valueRowSize,
                                                                                size_t maxRows)
      : bucketSize_(buckets),
        bucketCapacity_(bucketSize_),
        keyRowSize_(keyRowSize),
        valueRowSize_(valueRowSize),
        maxRows_(maxRows) {
    buckets_ = ptrAlloc_.allocate(bucketCapacity_);
    keyRows_.push_back(keyAlloc_.allocate(keyRowSize_));
    valueRows_.push_back(valueAlloc_.allocate(valueRowSize_));
    clear();
  }
  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::~SimpleAllocHashMultiMap() {
    for (typename std::list<KeyItem *>::iterator it = keyRows_.begin(), last = keyRows_.end(); it != last; ++it) {
      keyAlloc_.deallocate(*it, keyRowSize_);
    }
    for (typename std::list<ValueItem *>::iterator it = valueRows_.begin(), last = valueRows_.end(); it != last; ++it) {
      valueAlloc_.deallocate(*it, valueRowSize_);
    }
    ptrAlloc_.deallocate(buckets_, bucketCapacity_);
  }

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  void SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::freeRows() {
    if (keyRows_.size() > maxRows_) {
      //std::cerr << "Freeing key rows, current size is " << keyRows_.size() << std::endl;
      typename std::list<KeyItem *>::iterator it = keyRows_.begin(), last = keyRows_.end();
      for (std::advance(it, maxRows_); it != last; ++it) {
        keyAlloc_.deallocate(*it, keyRowSize_);
      }
      keyRows_.resize(maxRows_);
    }
    if (valueRows_.size() > maxRows_) {
      //std::cerr << "Freeing value rows, current size is " << valueRows_.size() << std::endl;
      typename std::list<ValueItem *>::iterator it = valueRows_.begin(), last = valueRows_.end();
      for (std::advance(it, maxRows_); it != last; ++it) {
        valueAlloc_.deallocate(*it, valueRowSize_);
      }
      valueRows_.resize(maxRows_);
    }
  }

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  typename SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::KeyItem *
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::push_back_(K const &key, KeyItem *next) {
    if (nextKeyItem_ == keyEndMarker_) {
      ++currentKeyRow_;
      if (currentKeyRow_ == keyRows_.end()) {
        keyRows_.push_back(keyAlloc_.allocate(keyRowSize_));
        currentKeyRow_ = keyRows_.end();
        --currentKeyRow_;  // end - 1 doesn't work!
      }
      nextKeyItem_ = *currentKeyRow_;
      keyEndMarker_ = nextKeyItem_ + keyRowSize_;
    }
    std::allocator_traits<KeyItemAllocator>::construct(keyAlloc_, nextKeyItem_, KeyItem(next, key, nullptr));
    nextKeyItem_++;
    return (nextKeyItem_ - 1);
  }
  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  typename SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::ValueItem *
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::push_back_(value_ref value, ValueItem *next) {
    if (nextValueItem_ == valueEndMarker_) {
      ++currentValueRow_;
      if (currentValueRow_ == valueRows_.end()) {
        valueRows_.push_back(valueAlloc_.allocate(valueRowSize_));
        currentValueRow_ = valueRows_.end();
        --currentValueRow_;  // end - 1 doesn't work!
      }
      nextValueItem_ = *currentValueRow_;
      valueEndMarker_ = nextValueItem_ + valueRowSize_;
    }
    std::allocator_traits<ValueItemAllocator>::construct(valueAlloc_, nextValueItem_, ValueItem(next, value));
    nextValueItem_++;
    return (nextValueItem_ - 1);
  }

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  typename SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::KeyItem &
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::find_or_insert_(K const &key) {
    //std::cout << "Find or insert for key " << key << std::endl;
    size_t hash = hasher_(key);
    KeyItem *&buck = buckets_[hash % bucketSize_];
    KeyItem *curr = buck;
    while (curr) {
      if (eq_(curr->key, key)) {
        //std::cout << "  Key " << key << " was found." << std::endl;
        return *curr;
      }
      curr = curr->next;
    }
    buck = push_back_(key, buck);
    return *buck;
  }

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  void SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::insert(K const &key, value_ref value) {
    //std::cout << "Pushing back value " << value << " for key " << key << std::endl;
    KeyItem &k = find_or_insert_(key);
    //std::cout << "Key " << (k.value ? "exists" : " is new") << std::endl;
    k.value = push_back_(value, k.value);
  }

  template <typename K, typename V, typename Hasher, typename Equals, typename Alloc>
  typename SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::value_iterator
  SimpleAllocHashMultiMap<K, V, Hasher, Equals, Alloc>::values(K const &key) {
    //std::cout << "Gettinv values for key " << key << std::endl;
    size_t hash = hasher_(key);
    for (KeyItem *curr = buckets_[hash % bucketSize_]; curr; curr = curr->next) {
      if (eq_(curr->key, key))
        return value_iterator(curr->value);
    }
    return value_iterator();
  }

  /*** Very very simple map implementation
 *   It's just a std::vector<pair<key,value>>, and the operator[] does a linear search to find the key (it's O(N) time, both if the key exists and if it doesn't)
 *   Anyway, if your map is very small and if you clear it often, it performs better than more complex variants
 *   Semantics is as std::map, except that only very few methods are implemented.
 * */
  template <typename K, typename V>
  class UnsortedDumbVectorMap {
  public:
    typedef std::pair<K, V> value_type;
    typedef typename std::vector<value_type>::const_iterator const_iterator;
    typedef typename std::vector<value_type>::iterator iterator;  // but please don't mutate the keys

    void clear() { data_.clear(); }
    bool empty() const { return data_.empty(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }

    V &operator[](const K &k) {
      for (typename std::vector<value_type>::iterator it = data_.begin(), ed = data_.end(); it != ed; ++it) {
        if (it->first == k)
          return it->second;
      }
      data_.push_back(value_type(k, V()));
      return data_.back().second;
    }

    UnsortedDumbVectorMap() {}

  private:
    std::vector<value_type> data_;
  };

}  // namespace cmsutil

#endif
