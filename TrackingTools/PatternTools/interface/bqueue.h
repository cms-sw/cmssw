#ifndef CMSUTILS_BEUEUE_H
#define CMSUTILS_BEUEUE_H
#include <boost/shared_ptr.hpp>

/**  Backwards linked queue with "head sharing"

     Author: Giovanni Petrucciani
     
     For use in trajectory building, where we want to "fork" a trajectory candidate in two
     without copying around all the hits before the fork.

     Supported operations (mimics a std container):
       - push_back()
       - fork() which returns a new queue that shares the head part with the old one
       - copy constructor, which just does fork()
       - rbegin(), rend(): return iterators which can work only backwards
            for (cmsutils::bqueue<T>::const_iterator it = queue.rbegin(), end = queue.rend();
                        it != end; --it) { ... }
       - front() and back(): returns a reference to the first and last element
       - size(), empty(): as expected. size() does not count the items
     
     Note that boost::shared_ptr is used for items, so they are deleted automatically
     while avoiding problems if one deletes a queue which shares the head with another one

     Disclaimer: I'm not sure the const_iterator is really const-correct..
*/
namespace cmsutils {

template<class T> class _bqueue_itr;
template<class T> class bqueue;

template <class T> 
class _bqueue_item {
        friend class bqueue<T>;
        friend class _bqueue_itr<T>;
    private:
        _bqueue_item() : back(0), value() { }
        _bqueue_item(boost::shared_ptr< _bqueue_item<T> > tail, const T &val) : back(tail), value(val) { }
        boost::shared_ptr< _bqueue_item<T> > back;
        T value;
};
template<class T>
class _bqueue_itr {
    public:
        T* operator->() { return &it->value; }
        T& operator*() { return it->value; }
        const T* operator->() const { return &it->value; }
        const T& operator*() const { return it->value; }
        _bqueue_itr<T> & operator--() { it = it->back; return *this; }
        const _bqueue_itr<T> & operator--() const { it = it->back.get(); return *this; }
        bool operator==(const _bqueue_itr<T> &t2) const  { return t2.it == it; }
        bool operator!=(const _bqueue_itr<T> &t2) const { return t2.it != it; }
        // so that I can assign a const_iterator to a const_iterator
        const _bqueue_itr<T> & operator=(const _bqueue_itr<T> &t2) const { it = t2.it; return *this; }
        friend class bqueue<T>;
    private:
        _bqueue_itr(_bqueue_item<T> *t) : it(t) { }
        _bqueue_itr(const _bqueue_item<T> *t) : it(t) { }
        mutable _bqueue_item<T> *it;
};

template<class T>
class bqueue {
    public:
        typedef T value_type;
        typedef unsigned short int size_type;
        typedef _bqueue_item<value_type>       item;
        typedef boost::shared_ptr< _bqueue_item<value_type> >  itemptr;
        typedef _bqueue_itr<value_type>       iterator;
        typedef const _bqueue_itr<value_type> const_iterator;
        bqueue() : m_size(0), m_bound(), m_head(m_bound), m_tail(m_bound) { }
        ~bqueue() { }
        bqueue(const bqueue<T> &cp) : m_size(cp.m_size), m_bound(cp.m_bound), m_head(cp.m_head), m_tail(cp.m_tail) { }
        bqueue<T> fork() {
            return bqueue<T>(m_size,m_bound,m_head,m_tail);
        }
        void push_back(const T& val) {
            m_tail = itemptr(new item(this->m_tail, val)); 
            if ((++m_size) == 1) { m_head = m_tail; };
        }
        void pop_back() {
            assert(m_size > 0);
            --m_size;
            m_tail = m_tail->back;
            if (m_size == 0) m_head = m_bound; 
        }
        T & front() { return m_head->value; }
        const T & front() const { return m_head->value; }
        T & back() { return m_tail->value; }
        const T & back() const { return m_tail->value; }
        iterator rbegin() { return m_tail.get(); }
        const_iterator rbegin() const { return m_tail.get(); }
        const_iterator rend() const { return m_bound.get(); }
        size_type size() const { return m_size; }
        bool empty() const { return m_size == 0; }
        const T & operator[](size_type i) const {
                int idx = m_size - i - 1;
                const_iterator it = rbegin();
                while (idx-- > 0) --it;
                return *it;
        }
    private:
        bqueue(size_type size, itemptr bound, itemptr head, itemptr tail) :
            m_size(size), m_bound(bound), m_head(head), m_tail(tail) { }
        size_type m_size;
        itemptr m_bound, m_head, m_tail;
        
};


}

#endif
