#ifndef TNT_ORTHOGONAL_PROJECTION_IMPL_HPP
#define TNT_ORTHOGONAL_PROJECTION_IMPL_HPP

#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>
#include <tnt/linear/orthogonal_projection.hpp>

#define VECTOR_DIM 1
#define ROW_INDEX 0

namespace tnt {

    namespace detail {

        template<typename DataType>

        struct OptimizedOrthogonalProjection<DataType> {

            static Tensor<DataType> eval(Tensor<DataType> &V1, Tensor<DataType> &V2) {
                DataType* v = V1.data.data;
                DataType* s = V2.data.data;
                int num_rows = V1.shape.axes[ROW_INDEX];

                scalar_multiply(s, (dotProduct(v, s, num_rows) / dotProduct(s, s, num_rows)), num_rows);

                tnt::AlignedPtr<DataType> ptr(s, num_rows);
                tnt::Tensor<DataType> proj(tnt::Shape{num_rows, VECTOR_DIM}, ptr);

                return proj;
            }

        private:

            static void scalar_multiply(DataType *data, DataType scalar, int num_rows) {
                for (int i = 0; i < num_rows; i++) { data[i] = data[i] * scalar; }
            }

            static float dotProduct(DataType *V1, DataType* V2, int num_rows) {
                float product = 0;
                for (int i = 0; i < num_rows; i++) { product += V1[i] * V2[i]; }
                return product;
            }

        };

    } // namespace detail

} // namespace tnt

using projection_data_types = doctest::Types<float>;

TEST_CASE_TEMPLATE("orthogonal_projection(const Tensor<T>&, const Tensor<T>&)", T, projection_data_types) {

    T v1Data[2] = {(T) 1, (T) 0};
    T v2Data[2] = {(T) 2, (T) 3};
    T v3Data[2] = {(T) 2, (T) 0};
    tnt::Tensor<T> v1(tnt::Shape{2, 1}, tnt::AlignedPtr<T>(v1Data, 2));
    tnt::Tensor<T> v2(tnt::Shape{2, 1}, tnt::AlignedPtr<T>(v2Data, 2));
    tnt::Tensor<T> v3(tnt::Shape{2, 1}, tnt::AlignedPtr<T>(v3Data, 2));

    REQUIRE(project(v2, v1) == v3);
}

TEST_CASE_TEMPLATE("orthogonal_projection(const Tensor<T>&, const Tensor<T>&)", T, projection_data_types) {

    T v1Data[3] = {(T) 1, (T) 2, (T) 0};
    T v2Data[3] = {(T) 1, (T) 1, (T) 2};
    T v3Data[3] = {(T) 1/2, (T) 1/2, (T) 1};
    T v4Data[3] = {(T) 3/5, (T) 6/5, (T) 0};
    tnt::Tensor<T> v1(tnt::Shape{3, 1}, tnt::AlignedPtr<T>(v1Data, 3));
    tnt::Tensor<T> v2(tnt::Shape{3, 1}, tnt::AlignedPtr<T>(v2Data, 3));
    tnt::Tensor<T> v3(tnt::Shape{3, 1}, tnt::AlignedPtr<T>(v3Data, 3));
    tnt::Tensor<T> v4(tnt::Shape{3, 1}, tnt::AlignedPtr<T>(v4Data, 3));

    REQUIRE(project(v1, v2) == v3);
    REQUIRE(project(v2, v1) == v4);

}

#endif // TNT_ORTHOGONAL_PROJECTION_IMPL_HPP
