#ifndef TNT_GAUSSIAN_ELIMINATION_HPP
#define TNT_GAUSSIAN_ELIMINATION_HPP

#include <tnt/core/tensor.hpp>
#include <type_traits>

namespace tnt
{

namespace detail
{

    template <typename DataType, typename Enable = void>
    struct OptimizedGaussianElimination
    {
        static Tensor<DataType> eval(const Tensor<DataType>&);
        static Tensor<DataType> eval(const Tensor<DataType>&, const Tensor<DataType>&);
    };

} // namespace detail

/// \brief Executes Gaussian Elimination of a Matrix
///
/// \param T1 The tensor. It has size `MxN`
/// \returns A tensor in Reduced Row Echelon Form
/// \requires T1 shall be two dimensional

template <typename DataType>
inline Tensor<DataType> gaussian_elim(Tensor<DataType>& T1)
{
    static_assert(std::is_floating_point<DataType>::value,
                  "eigenvalues() requires signed data");

    TNT_ASSERT(T1.shape.num_axes() == 2, InvalidParameterException("tnt::gaussian_elim()",
                               __FILE__, __LINE__, "Gaussian elimination requires 2D tensors"))

return detail::OptimizedGaussianElimination<DataType>::eval(T1);
}

template <typename DataType>
inline Tensor<DataType> gaussian_elim(Tensor<DataType>& T1, Tensor<DataType>& V1)
{
    static_assert(std::is_floating_point<DataType>::value,
                  "eigenvalues() requires signed data");

    TNT_ASSERT(T1.shape.num_axes() == 2, InvalidParameterException("tnt::gaussian_elim()",
                                                                       __FILE__, __LINE__, "Gaussian elimination requires 2D tensors"))

    TNT_ASSERT(V1.shape.axes[1] == 1, InvalidParameterException("tnt::gaussian_elim()",
                                                                    __FILE__, __LINE__, "Solution vector requires vector"));

    return detail::OptimizedGaussianElimination<DataType>::eval(T1, V1);
}

} // namespace tnt

#endif // TNT_GAUSSIAN_ELIMINATION_HPP