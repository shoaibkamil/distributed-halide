#ifndef _MPI_TIMING_H
#define _MPI_TIMING_H

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <limits>
#include <vector>
#include <time.h>
#include "mpi.h"

using std::numeric_limits;
using std::vector;

class MPITiming {
public:
    typedef enum {Min, Max, Median, Mean} Stat;
    typedef double timing_t;
    static const int MPI_TIMING_T = MPI_DOUBLE;

    MPITiming(MPI_Comm c) : _comm(c) {
        MPI_Comm_rank(_comm, &_rank);
        MPI_Comm_size(_comm, &_numprocs);
        _usempi = true;
        _reduced = -1.0;
        _percentile_lower = -1.0;
        _percentile_upper = -1.0;
        _gathered = -1.0;
        _gathered_percentile_lower = -1.0;
        _gathered_percentile_upper = -1.0;
    }

    MPITiming() {
        //tests();
        _rank = 0;
        _numprocs = 1;
        _usempi = false;
        _reduced = -1.0;
        _percentile_lower = -1.0;
        _percentile_upper = -1.0;
        _gathered = -1.0;
        _gathered_percentile_lower = -1.0;
        _gathered_percentile_upper = -1.0;
    }

    inline void barrier() const {
        MPI_Barrier(_comm);
    }

    void report() const {
        if (_rank == 0) {
            printf("Timing: <%d> ranks <%f> seconds, 20/80 percentile <%f,%f>\n",
                   _numprocs, gathered(), _gathered_percentile_lower, _gathered_percentile_upper);
        }
    }

    void nondistributed_report() const {
        if (_rank == 0) {
            printf("Timing: non-distributed <%f> seconds, 20/80 percentile <%f,%f>\n",
                   _reduced, _percentile_lower, _percentile_upper);
        }
    }

    void start() {
        clock_gettime(CLOCK_MONOTONIC, &_start_time);
    }

    timing_t stop() {
        clock_gettime(CLOCK_MONOTONIC, &_stop_time);
        timing_t sec = (_stop_time.tv_sec - _start_time.tv_sec) + (_stop_time.tv_nsec - _start_time.tv_nsec) / 1e9;
        return sec;
    }

    void record(timing_t sec) {
        _timings.push_back(sec);
    }

    // Reduce all recorded values by the given statistic.
    timing_t reduce(Stat stat) {
        _percentile_lower = compute_percentile(_timings, 20);
        _percentile_upper = compute_percentile(_timings, 80);
        _reduced = compute_stat(_timings, stat);
        //_timings.clear();
        return _reduced;
    }

    // Gather reduced values from all ranks to rank 0, and reduce the
    // gathered values by the given statistic.
    timing_t gather(timing_t value, Stat stat) {
        assert(_usempi);
        const int tag = 0;
        if (_rank == 0) {
            vector<timing_t> results;
            vector<timing_t> percentilelower, percentileupper;
            results.push_back(value);
            percentilelower.push_back(_percentile_lower);
            percentileupper.push_back(_percentile_upper);
            for (int i = 1; i < _numprocs; i++) {
                timing_t t;
                MPI_Recv(&t, 1, MPI_TIMING_T, i, tag, _comm, MPI_STATUS_IGNORE);
                results.push_back(t);
                MPI_Recv(&t, 1, MPI_TIMING_T, i, tag, _comm, MPI_STATUS_IGNORE);
                percentilelower.push_back(t);
                MPI_Recv(&t, 1, MPI_TIMING_T, i, tag, _comm, MPI_STATUS_IGNORE);
                percentileupper.push_back(t);
            }
            _gathered = compute_stat(results, stat);
            _gathered_percentile_lower = compute_stat(percentilelower, Min);
            _gathered_percentile_upper = compute_stat(percentileupper, Max);
        } else {
            MPI_Send(&value, 1, MPI_TIMING_T, 0, tag, _comm);
            MPI_Send(&_percentile_lower, 1, MPI_TIMING_T, 0, tag, _comm);
            MPI_Send(&_percentile_upper, 1, MPI_TIMING_T, 0, tag, _comm);
        }
        return _gathered;
    }

    // Gather reduced values from all ranks to rank 0, and reduce the
    // gathered values by the given statistic.
    timing_t gather(Stat stat) {
        return gather(_reduced, stat);
    }

    // Return the gathered result.
    timing_t gathered() const {
        assert(_usempi);
        return _gathered;
    }

    timing_t compute_percentile(int percentile) const {
        return compute_percentile(_timings, percentile);
    }
private:
    bool _usempi;
    int _rank, _numprocs;
    MPI_Comm _comm;
    vector<timing_t> _timings;
    timing_t _reduced;
    timing_t _percentile_lower, _percentile_upper;
    timing_t _gathered;
    timing_t _gathered_percentile_lower, _gathered_percentile_upper;
    struct timespec _start_time, _stop_time;

    timing_t compute_stat(const vector<timing_t> &values, Stat stat) const {
        switch (stat) {
        case Min:
            return compute_min(values);
        case Max:
            return compute_max(values);
        case Median:
            return compute_median(values);
        case Mean:
            return compute_mean(values);
        }
        assert(false && "Unknown statistic.");
        return 0.0;
    }

    timing_t compute_min(const vector<timing_t> &values) const {
        timing_t min = numeric_limits<timing_t>::max();
        for (timing_t v : values) {
            min = v <= min ? v : min;
        }
        return min;
    }

    timing_t compute_max(const vector<timing_t> &values) const {
        timing_t max = -numeric_limits<timing_t>::max();
        for (timing_t v : values) {
            max = v >= max ? v : max;
        }
        return max;
    }

    timing_t compute_median(const vector<timing_t> &values) const {
        timing_t median = 0;
        vector<timing_t> tmp(values.begin(), values.end());
        int midpoint = tmp.size() / 2;
        std::sort(tmp.begin(), tmp.end());
        if (tmp.size() % 2 == 0) {
            median = (tmp[midpoint - 1] + tmp[midpoint]) / 2;
        } else {
            median = tmp[midpoint];
        }
        return median;
    }

    timing_t compute_mean(const vector<timing_t> &values) const {
        timing_t sum = 0;
        for (timing_t v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    timing_t compute_percentile(const vector<timing_t> &values, int percentile) const {
        assert(0 < percentile && percentile <= 100);
        if (values.size() == 1) return values[0];
        vector<timing_t> tmp(values.begin(), values.end());
        std::sort(tmp.begin(), tmp.end());
        if (percentile == 100) {
            return tmp[tmp.size() - 1];
        }
        timing_t index = (percentile / 100.0) * (tmp.size() - 1);
        int lower = floor(index), upper = ceil(index);
        assert(0 <= lower && lower < (int)tmp.size() &&
               0 <= upper && upper < (int)tmp.size());
        if (lower == upper) {
            return tmp[upper];
        } else {
            timing_t frac, integ;
            frac = modf(index, &integ);
            return tmp[lower] + (tmp[upper] - tmp[lower]) * frac;
        }
    }

    bool eq(timing_t a, timing_t b) const {
        const timing_t thresh = 0.00001;
        return a == b || (std::abs(a - b) / b) < thresh;
    }

    void tests() const {
        vector<timing_t> values({5, 2, -16, 104, 22});
        assert(eq(compute_min(values), -16));
        assert(eq(compute_max(values), 104));
        assert(eq(compute_mean(values), 23.4));
        assert(eq(compute_median(values), 5));
        values.push_back(12);
        assert(eq(compute_median(values), 8.5));

        values = {0.003239, 0.003759, 0.002697, 0.001911, 0.004127, 0.003448, 0.001934, 0.002531, 0.001885, 0.002112};
        assert(eq(compute_min(values), 0.001885));
        assert(eq(compute_max(values), 0.004127));
        assert(eq(compute_mean(values), 0.0027643));
        assert(eq(compute_median(values), 0.002614));
        assert(eq(compute_percentile(values, 20), 0.0019294));
        assert(eq(compute_percentile(values, 50), 0.002614));
        assert(eq(compute_percentile(values, 80), 0.0035102));
    }
};

#endif
