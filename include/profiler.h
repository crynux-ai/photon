#pragma once

#include <map>

struct Metric {
    int obj_id;
    std::string tag;
    int64_t begin_ns;
    int64_t end_ns;
    int cnt;
};

class Profiler {

public: 
    static int64_t now_ns();
    static void begin(int obj_id, const std::string& tag);
    static void end(int obj_id, const std::string& tag);
    static void add(int obj_id, const std::string& tag, int64_t begin_ns, int64_t end_ns);

    static void print();
    static void clear();
    static std::map<int, std::map<std::string, std::vector<Metric>>> metrics;
};


#ifdef ENABLE_PROFILE
    #define PROFILE_BEGIN(obj_id, tag) Profiler::begin(obj_id, tag);
    #define PROFILE_END(obj_id, tag) Profiler::end(obj_id, tag);
    #define PROFILE_ADD(obj_id, tag, begin, end) Profiler::add(obj_id, tag, begin, end);
    #define PROFILE_PRINT Profiler::print(); Profiler::clear();
    #define PROFILE_TAG(tag) , tag
#else
    #define PROFILE_BEGIN(obj_id, tag)
    #define PROFILE_END(obj_id, tag)
    #define PROFILE_ADD(obj_id, tag, begin, end)
    #define PROFILE_PRINT
    #define PROFILE_TAG(tag)
#endif

