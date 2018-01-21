#include <benchmark/benchmark.h>

#include <tnt/core/core.hpp>
#include <tnt/math/math.hpp>

#include <opencv2/core.hpp>
#include <benchmark/opencv_utils.hpp>

#include <Eigen/Dense>

template <typename DataType>
static void add_TNT(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming(); // We don't count tensor creation
        tnt::Shape shape{size, size};

        tnt::Tensor<DataType, tnt::CPU> left(shape, 1.f);
        tnt::Tensor<DataType, tnt::CPU> right(shape, 2.f);

        state.ResumeTiming();
        tnt::add(left, right);
    }
}

template <typename DataType>
static void add_OCV(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        cv::Mat left  = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(1));
        cv::Mat right = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(2));

        cv::Mat dst;

        state.ResumeTiming();
        cv::add(left, right, dst);
    }
}

template <typename DataType>
static void add_EIG(benchmark::State& state, int size)
{
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    while (state.KeepRunning()) {
        state.PauseTiming();

        MatType left = MatType::Constant(size, size, 1);
        MatType right = MatType::Constant(size, size, 2);

        state.ResumeTiming();
        MatType dst = left + right;
    }
}

/*template <typename DataType>
static void add_CUS(benchmark::State& state)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        cv::Mat left = create_cv_mat<DataType>(state.range(0), state.range(1), cv::Scalar(1));
        cv::Mat right = create_cv_mat<DataType>(state.range(0), state.range(1), cv::Scalar(2));
        cv::Mat dst(state.range(0), state.range(1), CV_32FC1);

        const DataType* left_p = left.ptr<DataType>();
        const DataType* right_p = right.ptr<DataType>();
        DataType* dst_p = dst.ptr<DataType>();

        int width = state.range(0);
        int height = state.range(1);
        int total = width * height;

        state.ResumeTiming();
        int x = 0;
        for ( ; total--; x++)
            dst_p[x] = left_p[x] + right_p[x];
    }
}*/

template <typename T>
class RegisterAddBenchmark
{
public:
    RegisterAddBenchmark(const std::string& type)
    {
        std::vector<int> sizes{4, 16, 64, 512, 2048};
        for (int size : sizes) {
            std::string suffix = type + ">[" + std::to_string(size) + "x" + std::to_string(size) + "]";
            benchmark::RegisterBenchmark(("Add:TNT<" + suffix).c_str(), add_TNT<T>, size);
            benchmark::RegisterBenchmark(("Add:OCV<" + suffix).c_str(), add_OCV<T>, size);
            benchmark::RegisterBenchmark(("Add:EIG<" + suffix).c_str(), add_EIG<T>, size);
        }
    }
};

static RegisterAddBenchmark<int> add_benchmark_int("int");
static RegisterAddBenchmark<float> add_benchmark_float("float");
static RegisterAddBenchmark<double> add_benchmark_double("double");
