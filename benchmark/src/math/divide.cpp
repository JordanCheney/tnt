#include <benchmark/benchmark.h>

#include <tnt/core/core.hpp>
#include <tnt/math/math.hpp>

#include <opencv2/core.hpp>
#include <benchmark/opencv_utils.hpp>

#include <Eigen/Dense>

template <typename DataType>
static void divide_TNT(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming(); // We don't count tensor creation
        tnt::Shape shape{size, size};

        tnt::Tensor<DataType, tnt::CPU> left(shape, 1.f);
        tnt::Tensor<DataType, tnt::CPU> right(shape, 2.f);

        state.ResumeTiming();
        tnt::divide(left, right);
    }
}

template <typename DataType>
static void divide_OCV(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        cv::Mat left  = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(1));
        cv::Mat right = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(2));

        cv::Mat dst;

        state.ResumeTiming();
        cv::divide(left, right, dst);
    }
}

template <typename DataType>
static void divide_EIG(benchmark::State& state, int size)
{
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    while (state.KeepRunning()) {
        state.PauseTiming();

        MatType left = MatType::Constant(size, size, 1);
        MatType right = MatType::Constant(size, size, 2);

        state.ResumeTiming();
        MatType dst = left.cwiseQuotient(right);
    }
}

template <typename T>
class RegisterDivideBenchmark
{
public:
    RegisterDivideBenchmark(const std::string& type)
    {
        std::vector<int> sizes{4, 16, 64, 512, 2048};
        for (int size : sizes) {
            std::string suffix = type + ">[" + std::to_string(size) + "x" + std::to_string(size) + "]";
            benchmark::RegisterBenchmark(("Divide:TNT<" + suffix).c_str(), divide_TNT<T>, size);
            benchmark::RegisterBenchmark(("Divide:OCV<" + suffix).c_str(), divide_OCV<T>, size);
            benchmark::RegisterBenchmark(("Divide:EIG<" + suffix).c_str(), divide_EIG<T>, size);
        }
    }
};

static RegisterDivideBenchmark<int>    divide_benchmark_int("int");
static RegisterDivideBenchmark<float>  divide_benchmark_float("float");
static RegisterDivideBenchmark<double> divide_benchmark_double("double");
