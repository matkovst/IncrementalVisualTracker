#include "metrix.h"

using namespace std::chrono;

Metrix::Metrix() = default;
Metrix::~Metrix() = default;

void Metrix::start()
{
    m_start = steady_clock::now();
}

void Metrix::finish()
{
    m_finish = steady_clock::now();
    m_cumul += (m_finish - m_start);
    ++m_count;
}

std::int64_t Metrix::avgMilli() const
{
    return duration_cast<milliseconds>(m_cumul).count() / m_count;
}