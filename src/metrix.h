#pragma once

#include <chrono>

class Metrix final
{
public:
    Metrix();
    ~Metrix();

    void start();
    void finish();

    std::int64_t avgMilli() const;

private:
    std::int64_t m_count { 1 };
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_finish;
    std::chrono::steady_clock::duration m_cumul;
};