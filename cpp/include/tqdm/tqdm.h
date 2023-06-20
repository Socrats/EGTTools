#pragma once

/*
 *Copyright (c) 2018-2019 <Miguel Raggi> <mraggi@gmail.com>
 *
 *Permission is hereby granted, free of charge, to any person
 *obtaining a copy of this software and associated documentation
 *files (the "Software"), to deal in the Software without
 *restriction, including without limitation the rights to use,
 *copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the
 *Software is furnished to do so, subject to the following
 *conditions:
 *
 *The above copyright notice and this permission notice shall be
 *included in all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *OTHER DEALINGS IN THE SOFTWARE.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <type_traits>

// -------------------- chrono stuff --------------------

namespace tq
{
    using index = std::ptrdiff_t; // maybe std::size_t, but I hate unsigned types.
    using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

    inline double elapsed_seconds(time_point_t from, time_point_t to)
    {
        using seconds = std::chrono::duration<double>;
        return std::chrono::duration_cast<seconds>(to - from).count();
    }

    class Chronometer
    {
    public:
        Chronometer() : start_(std::chrono::steady_clock::now()) {}

        double reset()
        {
            auto previous = start_;
            start_ = std::chrono::steady_clock::now();

            return elapsed_seconds(previous, start_);
        }

        [[nodiscard]] double peek() const
        {
            auto now = std::chrono::steady_clock::now();

            return elapsed_seconds(start_, now);
        }

        [[nodiscard]] time_point_t get_start() const { return start_; }

    private:
        time_point_t start_;
    };

    // -------------------- progress_bar --------------------
    inline void clamp(double& x, double a, double b)
    {
        if (x < a) x = a;
        if (x > b) x = b;
    }

    class progress_bar
    {
    public:
        void restart()
        {
            chronometer_.reset();
            refresh_.reset();
        }

        void update(double progress)
        {
            clamp(progress, 0, 1);

            if (time_since_refresh() > min_time_per_update_ || progress == 0 ||
                progress == 1)
            {
                reset_refresh_timer();
                display(progress);
            }
            suffix_.str("");
        }

        void set_ostream(std::ostream& os) { os_ = &os; }
        void set_prefix(std::string s) { prefix_ = std::move(s); }
        void set_bar_size(int size) { bar_size_ = size; }
        void set_min_update_time(double time) { min_time_per_update_ = time; }

        template <class T>
        progress_bar& operator<<(const T& t)
        {
            suffix_ << t;
            return *this;
        }

        double elapsed_time() const { return chronometer_.peek(); }

    private:
        void display(double progress)
        {
            auto flags = os_->flags();

            double t = chronometer_.peek();
            double eta = t/progress - t;

            std::stringstream bar;

            bar << '\r' << prefix_ << '{' << std::fixed << std::setprecision(1)
                << std::setw(5) << 100*progress << "%} ";

            print_bar(bar, progress);

            bar << " (" << t << "s < " << eta << "s) ";

            std::string sbar = bar.str();
            std::string suffix = suffix_.str();

            index out_size = sbar.size() + suffix.size();
            term_cols_ = std::max(term_cols_, out_size);
            index num_blank = term_cols_ - out_size;

            (*os_) << sbar << suffix << std::string(num_blank, ' ') << std::flush;

            os_->flags(flags);
        }

        void print_bar(std::stringstream& ss, double filled) const
        {
            auto num_filled = static_cast<index>(std::round(filled*bar_size_));
            ss << '[' << std::string(num_filled, '#')
               << std::string(bar_size_ - num_filled, ' ') << ']';
        }

        double time_since_refresh() const { return refresh_.peek(); }
        void reset_refresh_timer() { refresh_.reset(); }

        Chronometer chronometer_{};
        Chronometer refresh_{};
        double min_time_per_update_{0.15}; // found experimentally

        std::ostream* os_{&std::cerr};

        index bar_size_{40};
        index term_cols_{1};

        std::string prefix_{};
        std::stringstream suffix_{};
    };

    // -------------------- iter_wrapper --------------------

    template <class ForwardIter, class Parent>
    class iter_wrapper
    {
    public:
        using iterator_category = typename ForwardIter::iterator_category;
        using value_type = typename ForwardIter::value_type;
        using difference_type = typename ForwardIter::difference_type;
        using pointer = typename ForwardIter::pointer;
        using reference = typename ForwardIter::reference;

        iter_wrapper(ForwardIter it, Parent* parent) : current_(it), parent_(parent)
        {}

        auto operator*() { return *current_; }

        void operator++() { ++current_; }

        template <class Other>
        bool operator!=(const Other& other) const
        {
            parent_->update(); // here and not in ++ because I need to run update
                              // before first advancement!
            return current_ != other;
        }

        bool operator!=(const iter_wrapper& other) const
        {
            parent_->update(); // here and not in ++ because I need to run update
                              // before first advancement!
            return current_ != other.current_;
        }

        [[nodiscard]] const ForwardIter& get() const { return current_; }

    private:
        friend Parent;
        ForwardIter current_;
        Parent* parent_;
    };

    // -------------------- tqdm_for_lvalues --------------------

    template <class ForwardIter, class EndIter = ForwardIter>
    class tqdm_for_lvalues
    {
    public:
        using this_t = tqdm_for_lvalues<ForwardIter, EndIter>;
        using iterator = iter_wrapper<ForwardIter, this_t>;
        using value_type = typename ForwardIter::value_type;
        using size_type = index;
        using difference_type = index;

        tqdm_for_lvalues(ForwardIter begin, EndIter end)
            : first_(begin, this), last_(end), num_iters_(std::distance(begin, end))
        {}

        tqdm_for_lvalues(ForwardIter begin, EndIter end, index total)
            : first_(begin, this), last_(end), num_iters_(total)
        {}

        template <class Container>
        explicit tqdm_for_lvalues(Container& C)
            : first_(C.begin(), this), last_(C.end()), num_iters_(C.size())
        {}

        template <class Container>
        explicit tqdm_for_lvalues(const Container& C)
            : first_(C.begin(), this), last_(C.end()), num_iters_(C.size())
        {}

        tqdm_for_lvalues(const tqdm_for_lvalues&) = delete;
        tqdm_for_lvalues(tqdm_for_lvalues&&) = delete;
        tqdm_for_lvalues& operator=(tqdm_for_lvalues&&) = delete;
        tqdm_for_lvalues& operator=(const tqdm_for_lvalues&) = delete;
        ~tqdm_for_lvalues() = default;

        template <class Container>
        tqdm_for_lvalues(Container&&) = delete; // prevent misuse!

        iterator begin()
        {
            bar_.restart();
            iters_done_ = 0;
            return first_;
        }

        EndIter end() const { return last_; }

        void update()
        {
            ++iters_done_;
            bar_.update(calc_progress());
        }

        void set_ostream(std::ostream& os) { bar_.set_ostream(os); }
        void set_prefix(std::string s) { bar_.set_prefix(std::move(s)); }
        void set_bar_size(int size) { bar_.set_bar_size(size); }
        void set_min_update_time(double time) { bar_.set_min_update_time(time); }

        template <class T>
        tqdm_for_lvalues& operator<<(const T& t)
        {
            bar_ << t;
            return *this;
        }

        void manually_set_progress(double to)
        {
            clamp(to, 0, 1);
            iters_done_ = std::round(to*num_iters_);
        }

    private:
        double calc_progress() const
        {
            double denominator = num_iters_;
            if (num_iters_ == 0) denominator += 1e-9;
            return iters_done_/denominator;
        }

        iterator first_;
        EndIter last_;
        index num_iters_{0};
        index iters_done_{0};
        progress_bar bar_;
    };

    template <class Container>
    tqdm_for_lvalues(Container&) -> tqdm_for_lvalues<typename Container::iterator>;

    template <class Container>
    tqdm_for_lvalues(const Container&)
            -> tqdm_for_lvalues<typename Container::const_iterator>;

    // -------------------- tqdm_for_rvalues --------------------

    template <class Container>
    class tqdm_for_rvalues
    {
    public:
        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;
        using value_type = typename Container::value_type;

        explicit tqdm_for_rvalues(Container&& C)
            : C_(std::forward<Container>(C)), tqdm_(C_)
        {}

        auto begin() { return tqdm_.begin(); }

        auto end() { return tqdm_.end(); }

        void update() { return tqdm_.update(); }

        void set_ostream(std::ostream& os) { tqdm_.set_ostream(os); }
        void set_prefix(std::string s) { tqdm_.set_prefix(std::move(s)); }
        void set_bar_size(int size) { tqdm_.set_bar_size(size); }
        void set_min_update_time(double time) { tqdm_.set_min_update_time(time); }

        template <class T>
        auto& operator<<(const T& t)
        {
            return tqdm_ << t;
        }

        void advance(index amount) { tqdm_.advance(amount); }

        void manually_set_progress(double to) { tqdm_.manually_set_progress(to); }

    private:
        Container C_;
        tqdm_for_lvalues<iterator> tqdm_;
    };

    template <class Container>
    tqdm_for_rvalues(Container &&) -> tqdm_for_rvalues<Container>;

    // -------------------- tqdm --------------------
    template <class ForwardIter>
    auto tqdm(const ForwardIter& first, const ForwardIter& last)
    {
        return tqdm_for_lvalues(first, last);
    }

    template <class ForwardIter>
    auto tqdm(const ForwardIter& first, const ForwardIter& last, index total)
    {
        return tqdm_for_lvalues(first, last, total);
    }

    template <class Container>
    auto tqdm(const Container& C)
    {
        return tqdm_for_lvalues(C);
    }

    template <class Container>
    auto tqdm(Container& C)
    {
        return tqdm_for_lvalues(C);
    }

    template <class Container>
    auto tqdm(Container&& C)
    {
        return tqdm_for_rvalues(std::forward<Container>(C));
    }

    // -------------------- int_iterator --------------------

    template <class IntType>
    class int_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = IntType;
        using difference_type = IntType;
        using pointer = IntType*;
        using reference = IntType&;

        explicit int_iterator(IntType val) : value_(val) {}

        IntType& operator*() { return value_; }

        int_iterator& operator++()
        {
            ++value_;
            return *this;
        }
        int_iterator& operator--()
        {
            --value_;
            return *this;
        }

        int_iterator& operator+=(difference_type d)
        {
            value_ += d;
            return *this;
        }

        difference_type operator-(const int_iterator& other) const
        {
            return value_ - other.value_;
        }

        bool operator!=(const int_iterator& other) const
        {
            return value_ != other.value_;
        }

    private:
        IntType value_;
    };

    // -------------------- range --------------------
    template <class IntType>
    class range
    {
    public:
        using iterator = int_iterator<IntType>;
        using const_iterator = iterator;
        using value_type = IntType;

        range(IntType first, IntType last) : first_(first), last_(last) {}
        explicit range(IntType last) : first_(0), last_(last) {}

        [[nodiscard]] iterator begin() const { return first_; }
        [[nodiscard]] iterator end() const { return last_; }
        [[nodiscard]] index size() const { return last_ - first_; }

    private:
        iterator first_;
        iterator last_;
    };

    template <class IntType>
    auto trange(IntType first, IntType last)
    {
        return tqdm(range(first, last));
    }

    template <class IntType>
    auto trange(IntType last)
    {
        return tqdm(range(last));
    }

    // -------------------- timing_iterator --------------------

    class timing_iterator_end_sentinel
    {
    public:
        explicit timing_iterator_end_sentinel(double num_seconds)
            : num_seconds_(num_seconds)
        {}

        [[nodiscard]] double num_seconds() const { return num_seconds_; }

    private:
        double num_seconds_;
    };

    class timing_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = double;
        using difference_type = double;
        using pointer = double*;
        using reference = double&;

        double operator*() const { return chrono_.peek(); }

        timing_iterator& operator++() { return *this; }

        bool operator!=(const timing_iterator_end_sentinel& other) const
        {
            return chrono_.peek() < other.num_seconds();
        }

    private:
        tq::Chronometer chrono_;
    };

    // -------------------- timer -------------------
    struct timer
    {
    public:
        using iterator = timing_iterator;
        using end_iterator = timing_iterator_end_sentinel;
        using const_iterator = iterator;
        using value_type = double;

        explicit timer(double num_seconds) : num_seconds_(num_seconds) {}

        [[nodiscard]] static iterator begin() { return iterator(); }
        [[nodiscard]] end_iterator end() const
        {
            return end_iterator(num_seconds_);
        }

        [[nodiscard]] double num_seconds() const { return num_seconds_; }

    private:
        double num_seconds_;
    };

    class tqdm_timer
    {
    public:
        using iterator = iter_wrapper<timing_iterator, tqdm_timer>;
        using end_iterator = timer::end_iterator;
        using value_type = typename timing_iterator::value_type;
        using size_type = index;
        using difference_type = index;

        explicit tqdm_timer(double num_seconds) : num_seconds_(num_seconds) {}

        tqdm_timer(const tqdm_timer&) = delete;
        tqdm_timer(tqdm_timer&&) = delete;
        tqdm_timer& operator=(tqdm_timer&&) = delete;
        tqdm_timer& operator=(const tqdm_timer&) = delete;
        ~tqdm_timer() = default;

        template <class Container>
        tqdm_timer(Container&&) = delete; // prevent misuse!

        iterator begin()
        {
            bar_.restart();
            return iterator(timing_iterator(), this);
        }

        end_iterator end() const { return end_iterator(num_seconds_); }

        void update()
        {
            double t = bar_.elapsed_time();

            bar_.update(t/num_seconds_);
        }

        void set_ostream(std::ostream& os) { bar_.set_ostream(os); }
        void set_prefix(std::string s) { bar_.set_prefix(std::move(s)); }
        void set_bar_size(int size) { bar_.set_bar_size(size); }
        void set_min_update_time(double time) { bar_.set_min_update_time(time); }

        template <class T>
        tqdm_timer& operator<<(const T& t)
        {
            bar_ << t;
            return *this;
        }

    private:
        double num_seconds_;
        progress_bar bar_;
    };

    inline auto tqdm(timer t) { return tqdm_timer(t.num_seconds()); }

} // namespace tq