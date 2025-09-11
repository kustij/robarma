#pragma once
#include <glog/logging.h>

namespace robarma
{
    // Helper to suppress Ceres/glog logging globally
    inline void disable_ceres_logging(const char *argv0 = "robarma")
    {
#ifndef ROBARMA_ENABLE_CERES_LOGGING
        static bool initialized = false;
        if (!initialized)
        {
            google::InitGoogleLogging(argv0);
            FLAGS_minloglevel = 3;
            FLAGS_logtostderr = 0;
            initialized = true;
        }
#endif
    }
}
