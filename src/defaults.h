#pragma once

#if DOUBLE_PRECISION == 0
    #define PRECISION float
    #define CV_PRECISION CV_32F
#else
    #define PRECISION double
    #define CV_PRECISION CV_64F
#endif