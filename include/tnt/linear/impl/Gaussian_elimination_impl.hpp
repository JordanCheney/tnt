#ifndef TNT_GAUSSIAN_ELIMINATION_IMPL_HPP
#define TNT_GAUSSIAN_ELIMINATION_IMPL_HPP

#include <tnt/utils/testing.hpp>
#include <tnt/utils/simd.hpp>
#include <tnt/linear/Gaussian_elimination.hpp>

namespace tnt {

    namespace detail {

        template<typename DataType>

        struct OptimizedGaussianElimination<DataType> {

            static Tensor<DataType> eval(Tensor<DataType> &tensor) {

                DataType *ptr = tensor.data.data;
                int num_rows = tensor.shape.axes[0];
                int num_cols = tensor.shape.axes[1];

                for (int i = 0; i < num_rows; i++) {
                    int lead = num_cols * i;
                    while (ptr[lead] == 0 && lead < (i*num_cols + num_rows)) {
                        lead++;
                    }

                    if (ptr[lead] == 1) {
                        clearColumn(ptr, i, (lead - (num_cols * i)), num_rows, num_cols);
                    } else if (lead != 0) {
                        divideRow(ptr, ptr[lead], i, num_cols);
                        clearColumn(ptr, i, (lead - (num_cols * i)), num_rows, num_cols);
                    }
                }
                naturalOrder(ptr, num_cols, num_rows);
                tensor.data = AlignedPtr<DataType>(ptr, tensor.shape.total());

                return tensor;
            }

        private:

            static int findLead(DataType* data, int num_cols, int row, int num_rows) {
                int lead_index = num_cols * row;
                while (data[lead_index] == 0 && lead_index < (row * num_cols + num_rows)) {
                    lead_index++;
                }
                return lead_index - (row * num_cols);
            }

            static void switchRows(DataType* data, int row1, int row2, int num_cols) {
                std::cout << "switching rows " << row1 << " & " << row2 << std::endl;
                for (int i = 0; i < num_cols; i++) {
                    int temp = data[row1 * num_cols + i];
                    data[row1 * num_cols + i] = data[row2 * num_cols + i];
                    data[row2 * num_cols + i] = temp;
                }

            }

            static void naturalOrder(DataType* data, int num_cols, int num_rows)
            {
                std::cout << "tensor is: " << std::endl;
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < num_cols; j++) {
                        std::cout << data[i*num_cols + j];
                    }
                    std::cout << "\n";
                }

                for (int i = 1; i < num_rows; i++) {
                    int j = i;
                    while (j >= 1 && (findLead(data, num_cols, j, num_rows) < findLead(data, num_cols, j - 1, num_rows))) {
                        switchRows(data, j, j - 1, num_cols);
                        j = j - 1;
                    }
                }
            }

            static void divideRow(DataType *data, DataType lead, int row, int num_cols)
            {
                for (int i = 0; i < num_cols; i++) {
                    if (data[(num_cols * row) + i] != 0)
                        data[(num_cols * row) + i] = data[(num_cols * row) + i] / lead;
                }
            }

            static void subtractRow(DataType *data, int row, DataType val, int num_cols, int base)
            {
                for (int i = 0; i < num_cols; i++) {
                    data[(num_cols * row) + i] = data[(num_cols * row) + i] - (val * (data[(num_cols * base) + i]));
                }
            }

            static void clearColumn(DataType *data, int row, int leading_index, int num_rows, int num_cols)
            {
                for (int i = 0; i < num_rows; i++) {
                    if (i != row) {
                        DataType match = data[(num_cols * i) + leading_index];
                        if (match != 0) {
                            subtractRow(data, i, match, num_cols, row);
                        }
                    }
                }
            }
        };

    } // namespace detail

} // namespace tnt

using multiply_data_types = doctest::Types<float>;

TEST_CASE_TEMPLATE("gaussian_elim(const Tensor<T>&)", T, multiply_data_types)
{
    T data[9] = {(T) 1, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 1};
    tnt::AlignedPtr<T> ptr(data, 9);
    tnt::Tensor<T> t1(tnt::Shape{3, 3}, ptr);
    T rref[9] = {(T) 1, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 1};
    tnt::AlignedPtr<T> ptr1(rref, 9);
    tnt::Tensor<T> t2(tnt::Shape{3, 3}, ptr1);

    REQUIRE(gaussian_elim(t1) == t2);

    T data1[9] = {(T) 1, (T) 2, (T) 3, (T) 2, (T) 0, (T) 1, (T) 1, (T) 0, (T) 1};
    tnt::AlignedPtr<T> ptr2(data1, 9);
    tnt::Tensor<T> t3(tnt::Shape{3, 3}, ptr2);

    REQUIRE(gaussian_elim(t3) == t2);

    T data3[9] = {(T) 1, (T) 0, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 1, (T) 0};
    tnt::AlignedPtr<T> ptr6(data3, 9);
    tnt::Tensor<T> t4(tnt::Shape{3, 3}, ptr6);

    REQUIRE(gaussian_elim(t4) == t2);

    T data2[9] = {(T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0};
    tnt::AlignedPtr<T> ptr4(data2, 9);
    tnt::Tensor<T> t5(tnt::Shape{3, 3}, ptr4);
    T rref2[9] = {(T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0};
    tnt::AlignedPtr<T> ptr5(rref2, 9);
    tnt::Tensor<T> t6(tnt::Shape{3, 3}, ptr5);

    REQUIRE(gaussian_elim(t5) == t6);

    T data5[9] = {(T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0};
    tnt::AlignedPtr<T> ptr7(data5, 9);
    tnt::Tensor<T> t9(tnt::Shape{3, 3}, ptr7);
    T rref3[9] = {(T) 1, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 0};
    tnt::AlignedPtr<T> ptr8(rref3, 9);
    tnt::Tensor<T> t10(tnt::Shape{3, 3}, ptr8);

    REQUIRE(gaussian_elim(t9) == t10);

    T data11[9] = {(T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 1, (T) 0};
    tnt::AlignedPtr<T> ptr11(data11, 9);
    tnt::Tensor<T> t11(tnt::Shape{3, 3}, ptr11);
    T rref11[9] = {(T) 0, (T) 1, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0, (T) 0};
    tnt::AlignedPtr<T> ptr12(rref11, 9);
    tnt::Tensor<T> t12(tnt::Shape{3, 3}, ptr12);

    REQUIRE(gaussian_elim(t11) == t12);

    T d[6] = {(T) 1, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0};
    tnt::AlignedPtr<T> p(d, 6);
    tnt::Tensor<T> tt(tnt::Shape{3, 2}, p);
    T r[6] = {(T) 1, (T) 0, (T) 0, (T) 1, (T) 0, (T) 0};
    tnt::AlignedPtr<T> p1(r, 6);
    tnt::Tensor<T> tt1(tnt::Shape{3, 2}, p1);

    REQUIRE(gaussian_elim(tt) == tt1);

}


#endif // TNT_GAUSSIAN_ELIMINATION_IMPL_HPP