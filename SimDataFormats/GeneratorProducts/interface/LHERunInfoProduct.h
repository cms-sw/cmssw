#ifndef SimDataFormats_GeneratorProducts_LHERunInfoProduct_h
#define SimDataFormats_GeneratorProducts_LHERunInfoProduct_h

#include <iterator>
#include <memory>
#include <vector>
#include <string>

//#include <hepml.hpp>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

class LHERunInfoProduct {
    public:
    typedef std::vector<std::pair<std::string,std::string> > weights_defs;
	class Header {
	    public:
		typedef std::vector<std::string>::const_iterator const_iterator;
		typedef std::vector<std::string>::size_type size_type;

		Header() {}
		Header(const std::string &tag) : tag_(tag) {}
		~Header() {}

		void addLine(const std::string &line) { lines_.push_back(line); }

		const std::string &tag() const { return tag_; }
		const std::vector<std::string> &lines() const { return lines_; }

		size_type size() const { return lines_.size(); }
		const_iterator begin() const { return lines_.begin(); }
		const_iterator end() const { return lines_.end(); }

		bool operator == (const Header &other) const
		{ return tag_ == other.tag_ && lines_ == other.lines_; }
		inline bool operator != (const Header &other) const
		{ return !(*this == other); }

	    private:
		std::string			tag_;
		std::vector<std::string>	lines_;
	};

	typedef std::vector<Header>::size_type size_type;
	typedef std::vector<Header>::const_iterator headers_const_iterator;
	typedef std::vector<std::string>::const_iterator
						comments_const_iterator;

	LHERunInfoProduct() {}
	LHERunInfoProduct(const lhef::HEPRUP &heprup) : heprup_(heprup) {}
	~LHERunInfoProduct() {}

	void addHeader(const Header &header) { headers_.push_back(header); }
	void addComment(const std::string &line) { comments_.push_back(line); }

	const lhef::HEPRUP &heprup() const { return heprup_; }

	size_type headers_size() const { return headers_.size(); }
	headers_const_iterator headers_begin() const { return headers_.begin(); }
	headers_const_iterator headers_end() const { return headers_.end(); }

	size_type comments_size() const { return comments_.size(); }
	comments_const_iterator comments_begin() const { return comments_.begin(); }
	comments_const_iterator comments_end() const { return comments_.end(); }

	class const_iterator {
	    public:
		typedef std::forward_iterator_tag	iterator_category;
		typedef std::string			value_type;
		typedef std::ptrdiff_t			difference_type;
		typedef std::string			*pointer;
		typedef std::string			&reference;

		const_iterator() : mode(kDone) {}
		~const_iterator() {}

		bool operator == (const const_iterator &other) const;
		inline bool operator != (const const_iterator &other) const
		{ return !operator == (other); }

		inline const_iterator &operator ++ ()
		{ next(); return *this; }
		inline const_iterator operator ++ (int dummy)
		{ const_iterator orig = *this; next(); return orig; }

		const std::string &operator * () const { return tmp; }
		const std::string *operator -> () const { return &tmp; }

	    private:
		friend class LHERunInfoProduct;

		void next();

		enum Mode {
			kHeader,
			kBody,
			kInit,
			kDone,
			kFooter
		};

		const LHERunInfoProduct	*runInfo;
		headers_const_iterator	header;
		Header::const_iterator	iter;
		Mode			mode;
		unsigned int		line;
		std::string		tmp;
	};

	const_iterator begin() const;
	const_iterator init() const;
	inline const_iterator end() const { return const_iterator(); }

	static const std::string &endOfFile();

	bool operator == (const LHERunInfoProduct &other) const
	{ return heprup_ == other.heprup_ && headers_ == other.headers_ && comments_ == other.comments_; }
	inline bool operator != (const LHERunInfoProduct &other) const
	{ return !(*this == other); }

	bool mergeProduct(const LHERunInfoProduct &other);
    	bool isProductEqual(const LHERunInfoProduct &other) const
    	{ return *this == other; }
        static bool isTagComparedInMerge(const std::string& tag);

    private:
	lhef::HEPRUP			heprup_;
	std::vector<Header>		headers_;
	std::vector<std::string>	comments_;
};

#endif // GeneratorRunInfo_LHEInterface_LHERunInfoProduct_h
