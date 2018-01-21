#include <tnt_benchmark_harness.hpp>

#include <tnt/types/tensor.hpp>

#include <opencv2/core.hpp>

namespace tnt_bench
{

// ----------------------------------------------------------------------------
// Run a TNT iteration at a specified optimization level

template <typename DataType, typename DeviceType, int optimization>
static inline uint64_t run_tnt_iteration(const tnt::Shape& shape)
{
    auto left  = tnt::Tensor<DataType, DeviceType>(shape, 1.f);
    auto right = tnt::Tensor<DataType, DeviceType>(shape, 2.f);

    auto start = now();
    tnt::add<DataType, DeviceType, optimization>(left, right);
    return elapsed(start);
}

// ----------------------------------------------------------------------------
// Run an OpenCV iteration

static inline uint64_t run_opencv_iteration(const tnt::Shape& shape, int type)
{

    auto left  = cv::Mat(shape[0], shape[1], type, cv::Scalar(1));
    auto right = cv::Mat(shape[0], shape[1], type, cv::Scalar(2));
    cv::Mat dst;

    auto start = now();
    cv::add(left, right, dst);
    return elapsed(start);
}

static inline uint64_t run_custom_iteration(const tnt::Shape& shape)
{
    cv::Mat left(shape[0], shape[1], CV_32S, cv::Scalar(1));
    cv::Mat right(shape[0], shape[1], CV_32S, cv::Scalar(2));
    cv::Mat dst(shape[0], shape[1], CV_32S);

    auto start = now();
    {
        const int* left_p = left.ptr<int>();
        size_t left_step = left.step;
        const int* right_p = right.ptr<int>();
        size_t right_step = right.step;
        int* dst_p = dst.ptr<int>();
        size_t dst_step = dst.step;
        int width = shape[0];
        int height = shape[1];

        for( ; height--; left_p  = (const int*) ((const uchar*) left_p  + left_step),
                         right_p = (const int*) ((const uchar*) right_p + right_step),
                         dst_p   = (int*) ((uchar*)dst_p + dst_step))
        {
            int x = 0;
            // My code
            for( ; x < width; x++ )
                dst_p[x] = left_p[x] + right_p[x];
        }
    }

    return elapsed(start);
}

// ----------------------------------------------------------------------------
// Add function benchmark

struct AddFunctionBenchmark : public Benchmark
{
    const std::string name() const
    {
        return "AddFunctionBenchmark";
    }

    std::vector<tnt::Shape> shapes() const
    {
        return {
            {4, 4},
            {16, 16},
            {64, 64},
            {512, 512},
            {2048, 2048}
        };
    }

    std::vector<BenchmarkResult> run_tnt(const std::vector<tnt::Shape>& shapes) const
    {
        std::vector<BenchmarkResult> results;

#define RUN_TNT_NO_OP(type, label)                                                                     \
{                                                                                                      \
    BenchmarkResult result("TNT", label, "None");                                                      \
    for (const tnt::Shape& shape : shapes)                                                             \
        result.times.push_back(run_tnt_iteration<type, tnt::CPU, tnt::instruction_sets::NONE>(shape)); \
    results.push_back(result);                                                                         \
}

      //RUN_TNT_NO_OP(int8_t,  "8S");
      //RUN_TNT_NO_OP(int16_t, "16S");
      RUN_TNT_NO_OP(int32_t, "32S");
      //RUN_TNT_NO_OP(int64_t, "64S");

      //RUN_TNT_NO_OP(uint8_t,  "8U");
      //RUN_TNT_NO_OP(uint16_t, "16U");
      //RUN_TNT_NO_OP(uint32_t, "32U");
      //RUN_TNT_NO_OP(uint64_t, "64U");

      //RUN_TNT_NO_OP(float, "32F");
      //RUN_TNT_NO_OP(double, "64F");

#undef RUN_TNT_NO_OP

#ifdef HAVE_SSE4

#define RUN_TNT_SSE4(type, label)                                                                      \
{                                                                                                      \
    BenchmarkResult result("TNT", label, "SSE4");                                                      \
    for (const tnt::Shape& shape : shapes)                                                             \
        result.times.push_back(run_tnt_iteration<type, tnt::CPU, tnt::instruction_sets::SSE4>(shape)); \
    results.push_back(result);                                                                         \
}

        //RUN_TNT_SSE4(int8_t,  "8S");
        //RUN_TNT_SSE4(int16_t, "16S");
        //RUN_TNT_SSE4(int32_t, "32S");
        //RUN_TNT_SSE4(int64_t, "64S");

        //RUN_TNT_SSE4(uint8_t,  "8U");
        //RUN_TNT_SSE4(uint16_t, "16U");
        //RUN_TNT_SSE4(uint32_t, "32U");
        //RUN_TNT_SSE4(uint64_t, "64U");

        //RUN_TNT_SSE4(float, "32F");
        //RUN_TNT_SSE4(double, "64F");

#undef RUN_TNT_SSE4

#endif

#ifdef HAVE_AVX

#define RUN_TNT_AVX(type, label)                                                                       \
{                                                                                                      \
    BenchmarkResult result("TNT", label, "AVX");                                                       \
    for (const tnt::Shape& shape : shapes)                                                             \
        result.times.push_back(run_tnt_iteration<type, tnt::CPU, tnt::instruction_sets::AVX>(shape));  \
    results.push_back(result);                                                                         \
}

        RUN_TNT_AVX(int8_t,  "8S");
        RUN_TNT_AVX(int16_t, "16S");
        RUN_TNT_AVX(int32_t, "32S");
        RUN_TNT_AVX(int64_t, "64S");

        RUN_TNT_AVX(uint8_t,  "8U");
        RUN_TNT_AVX(uint16_t, "16U");
        RUN_TNT_AVX(uint32_t, "32U");
        RUN_TNT_AVX(uint64_t, "64U");

        RUN_TNT_AVX(float, "32F");
        RUN_TNT_AVX(double, "64F");

#undef RUN_TNT_AVX

#endif

#ifdef HAVE_AVX2

#define RUN_TNT_AVX2(type, label)                                                                      \
{                                                                                                      \
    BenchmarkResult result("TNT", label, "AVX2");                                                      \
    for (const tnt::Shape& shape : shapes)                                                             \
        result.times.push_back(run_tnt_iteration<type, tnt::CPU, tnt::instruction_sets::AVX2>(shape)); \
    results.push_back(result);                                                                         \
}

        RUN_TNT_AVX2(int8_t,  "8S");
        RUN_TNT_AVX2(int16_t, "16S");
        RUN_TNT_AVX2(int32_t, "32S");
        RUN_TNT_AVX2(int64_t, "64S");

        RUN_TNT_AVX2(uint8_t,  "8U");
        RUN_TNT_AVX2(uint16_t, "16U");
        RUN_TNT_AVX2(uint32_t, "32U");
        RUN_TNT_AVX2(uint64_t, "64U");

        RUN_TNT_AVX2(float, "32F");
        RUN_TNT_AVX2(double, "64F");

#undef RUN_TNT_AVX2

#endif

        return results;
    }

    std::vector<BenchmarkResult> run_opencv(const std::vector<tnt::Shape>& shapes) const
    {
        std::vector<BenchmarkResult> results;

#define RUN_OPENCV_TYPE(type, label)                               \
{                                                                  \
    BenchmarkResult result("OCV", label, "Best");                  \
    for (const tnt::Shape& shape : shapes)                         \
        result.times.push_back(run_opencv_iteration(shape, type)); \
    results.push_back(result);                                     \
}

        //RUN_OPENCV_TYPE(CV_8S, "8S")
        //RUN_OPENCV_TYPE(CV_16S, "16S")
        RUN_OPENCV_TYPE(CV_32S, "32S")

        //RUN_OPENCV_TYPE(CV_8U, "8U")
        //RUN_OPENCV_TYPE(CV_16U, "16U")

        //RUN_OPENCV_TYPE(CV_32F, "32F")
        //RUN_OPENCV_TYPE(CV_64F, "64F")

#undef RUN_OPENCV_TYPE

#define RUN_CUSTOM_TYPE(type, label)                               \
{                                                                  \
    BenchmarkResult result("CUS", label, "Best");                  \
    for (const tnt::Shape& shape : shapes)                         \
        result.times.push_back(run_custom_iteration(shape));       \
    results.push_back(result);                                     \
}

        //RUN_OPENCV_TYPE(CV_8S, "8S")
        //RUN_OPENCV_TYPE(CV_16S, "16S")
        RUN_CUSTOM_TYPE(CV_32S, "32S")

        //RUN_OPENCV_TYPE(CV_8U, "8U")
        //RUN_OPENCV_TYPE(CV_16U, "16U")

        //RUN_OPENCV_TYPE(CV_32F, "32F")
        //RUN_OPENCV_TYPE(CV_64F, "64F")

#undef RUN_CUSTOM_TYPE

        return results;
    }
};

RegisterBenchmark<AddFunctionBenchmark> register_add_function_benchmark;

// ----------------------------------------------------------------------------

} // namespace tnt_bench

// ----------------------------------------------------------------------------
