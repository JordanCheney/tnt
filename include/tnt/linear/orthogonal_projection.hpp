#ifndef TNT_ORTHOGONAL_PROJECTION_HPP
#define TNT_ORTHOGONAL_PROJECTION_HPP

#include <tnt/core/tensor.hpp>
#include <type_traits>

namespace tnt
{

    namespace detail
    {

        template <typename DataType, typename Enable = void>
        struct OptimizedOrthogonalProjection
        {
            static Tensor<DataType> eval(const Tensor<DataType>&, const Tensor<DataType>&);
        };

    } // namespace detail

/// \brief Executes Orthogonal Projection of a Vector onto a line spanned by another vector
///
/// \param V1 The vector to be projected. It has size `Nx1'
/// \param V2 The vector which spans the line to be projeted upon
/// \returns The orthogonal projection of V1
/// \requires V1 and V2 shall be one dimensional

    template <typename DataType>
    inline Tensor<DataType> project(Tensor<DataType>& V1, Tensor<DataType> V2)
    {

        TNT_ASSERT(V2.shape.axes[1] == 1, InvalidParameterException("tnt::project()",
                                                                       __FILE__, __LINE__, "projection requires a vector"));

        TNT_ASSERT(V1.shape.axes[1] == 1, InvalidParameterException("tnt::project()",
                                                                       __FILE__, __LINE__, "projection requires a vector"));

        TNT_ASSERT(V2.shape.axes[0] == V1.shape.axes[0], InvalidParameterException("tnt::project()",
                                                                                   __FILE__, __LINE__, "vectors must have the same dimensions"));

        return detail::OptimizedOrthogonalProjection<DataType>::eval(V1, V2);
    }

} // namespace tnt

#endif // TNT_ORTHOGONAL_PROJECTION_HPP