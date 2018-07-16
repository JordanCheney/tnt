#include <benchmark/benchmark.h>

#include <tnt/core/core.hpp>
#include <tnt/math/math.hpp>

#include <opencv2/core.hpp>
#include <benchmark/opencv_utils.hpp>

#include <Eigen/Dense>
#include <blas/cblas.hpp>

template <typename DataType>
static void multiply_TNT(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming(); // We don't count tensor creation
        tnt::Shape shape{size, size};

        tnt::Tensor<DataType> left(shape, 3.f);
        tnt::Tensor<DataType> right(shape, 4.f);

        state.ResumeTiming();
        tnt::multiply(left, right);
    }
}

template <typename DataType>
static void multiply_OCV(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        cv::Mat left  = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(3));
        cv::Mat right = tnt::create_cv_mat<DataType>(size, size, cv::Scalar(4));

        state.ResumeTiming();
        cv::Mat dst = left * right;
    }
}

template <typename DataType>
static void multiply_EIG(benchmark::State& state, int size)
{
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    while (state.KeepRunning()) {
        state.PauseTiming();

        MatType left = MatType::Constant(size, size, 3);
        MatType right = MatType::Constant(size, size, 4);

        state.ResumeTiming();
        MatType dst = left * right;
    }
}

template <typename DataType>
struct BLASMultiply {};

template <> struct BLASMultiply<float>
{
    static void run(int size, float* left, float* right, float* dst)
    {
        /* See https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html */
        /* For trick to use fortran order (col-major) */
        cblas_sgemm(CblasColMajor, /* Memory order */
                    CblasNoTrans,  /* Left is transposed? */
                    CblasNoTrans,  /* Right is transposed? */
                    size,          /* Rows of left */
                    size,          /* Columns of right */
                    size,          /* Columns of left and rows of right */
                    1.,            /* Alpha (scale after mult) */
                    right,         /* Left data */
                    size,          /* lda, this is really for 2D mats */
                    left,          /* right data */
                    size,          /* ldb, this is really for 2D mats */
                    0.,            /* beta (scale on result before addition) */
                    dst,           /* result data */
                    size);         /* ldc, this is really for 2D mats */
    }
};

template <> struct BLASMultiply<double>
{
    static void run(int size, double* left, double* right, double* dst)
    {
        /* See https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html */
        /* For trick to use fortran order (col-major) */
        cblas_dgemm(CblasColMajor, /* Memory order */
                    CblasNoTrans,  /* Left is transposed? */
                    CblasNoTrans,  /* Right is transposed? */
                    size,          /* Rows of left */
                    size,          /* Columns of right */
                    size,          /* Columns of left and rows of right */
                    1.,            /* Alpha (scale after mult) */
                    right,         /* Left data */
                    size,          /* lda, this is really for 2D mats */
                    left,          /* right data */
                    size,          /* ldb, this is really for 2D mats */
                    0.,            /* beta (scale on result before addition) */
                    dst,           /* result data */
                    size);         /* ldc, this is really for 2D mats */
    }
};

template <typename DataType>
static void multiply_BLAS(benchmark::State& state, int size)
{
    while (state.KeepRunning()) {
        state.PauseTiming();

        DataType* left  = new DataType[size * size];
        DataType* right = new DataType[size * size];

        for (int i = 0; i < size * size; ++i) {
            left[i] = 3;
            right[i] = 4;
        }

        state.ResumeTiming();

        // For fairness allocation happens during timing
        DataType* dst = new DataType[size * size];

        BLASMultiply<DataType>::run(size, left, right, dst);
    }
}

template <typename T>
class RegisterMatrixMultiplyBenchmark
{
public:
    RegisterMatrixMultiplyBenchmark(const std::string& type)
    {
        std::vector<int> sizes{4, 16, 64, 512, 2048, 3121};
        for (int size : sizes) {
            std::string suffix = type + ">[" + std::to_string(size) + "x" + std::to_string(size) + "]";
            benchmark::RegisterBenchmark(("MatrixMultiply:TNT <" + suffix).c_str(), multiply_TNT<T>, size);
            benchmark::RegisterBenchmark(("MatrixMultiply:OCV <" + suffix).c_str(), multiply_OCV<T>, size);
            benchmark::RegisterBenchmark(("MatrixMultiply:EIG <" + suffix).c_str(), multiply_EIG<T>, size);
            benchmark::RegisterBenchmark(("MatrixMultiply:BLAS<" + suffix).c_str(), multiply_BLAS<T>, size);
        }
    }
};

//static RegisterMultiplyBenchmark<int>    multiply_benchmark_int("int");
static RegisterMatrixMultiplyBenchmark<float>  matrix_multiply_benchmark_float("float");
static RegisterMatrixMultiplyBenchmark<double> matrix_multiply_benchmark_double("double");
