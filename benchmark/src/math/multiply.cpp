#include <benchmark/benchmark.h>

#include <tnt/core/core.hpp>
#include <tnt/math/math.hpp>

#include <opencv2/core.hpp>
#include <benchmark/opencv_utils.hpp>

#include <Eigen/Dense>
#include <tnt/deps/cblas.hpp>

template <typename DataType>
static void multiply_TNT(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming(); // We don't count tensor creation
        tnt::Shape shape{size, size};

        tnt::Tensor<DataType> left(shape, 1.f);
        tnt::Tensor<DataType> right(shape, 2.f);

        state.ResumeTiming();
        tnt::multiply(left, right);
    }
}

template <typename DataType>
static void multiply_OCV(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        cv::Mat left  = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(1));
        cv::Mat right = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(2));

        cv::Mat dst;

        state.ResumeTiming();
        cv::multiply(left, right, dst);
    }
}

template <typename DataType>
static void multiply_EIG(benchmark::State& state, int size)
{
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    while (state.KeepRunning()) {
        state.PauseTiming();

        MatType left = MatType::Constant(size, size, 1);
        MatType right = MatType::Constant(size, size, 2);

        state.ResumeTiming();
        MatType dst = left.cwiseProduct(right);
    }
}

template <typename T>
class RegisterMultiplyBenchmark
{
public:
    RegisterMultiplyBenchmark(const std::string& type)
    {
        std::vector<int> sizes{4, 16, 64, 512, 2048};
        for (int size : sizes) {
            std::string suffix = type + ">[" + std::to_string(size) + "x" + std::to_string(size) + "]";
            benchmark::RegisterBenchmark(("Multiply:TNT<" + suffix).c_str(), multiply_TNT<T>, size);
            benchmark::RegisterBenchmark(("Multiply:OCV<" + suffix).c_str(), multiply_OCV<T>, size);
            benchmark::RegisterBenchmark(("Multiply:EIG<" + suffix).c_str(), multiply_EIG<T>, size);
        }
    }
};

static RegisterMultiplyBenchmark<int>    multiply_benchmark_int("int");
static RegisterMultiplyBenchmark<float>  multiply_benchmark_float("float");
static RegisterMultiplyBenchmark<double> multiply_benchmark_double("double");
