#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <algorithm>
#include <arm_sve.h>

template <typename T>
void sve_reduce_kernel_impl(T* dst, const T* src, size_t num_elements);



inline int64_t get_optimal_grain_size(size_t num_elements, size_t element_bytes) {
    int total_threads = at::get_num_threads(); // 获取全局线程数 (比如 320)
    

    int64_t max_effective_threads = std::min(total_threads, 16); 


    int64_t min_elements_per_thread = (256 * 1024) / element_bytes;
    int64_t target_threads = num_elements / min_elements_per_thread;
    

    target_threads = std::max<int64_t>(1, std::min<int64_t>(max_effective_threads, target_threads));


    int64_t grain_size = (num_elements + target_threads - 1) / target_threads;
    
    return std::max<int64_t>(256LL, grain_size);
}


template <>
void sve_reduce_kernel_impl<float>(float* dst, const float* src, size_t num_elements) {
    int64_t grain_size = get_optimal_grain_size(num_elements, sizeof(float));

    at::parallel_for(0, num_elements, grain_size, [&](int64_t start, int64_t end) {
        uint64_t vl = svcntw(); // Count Words (32位)
        uint64_t i = static_cast<uint64_t>(start);
        uint64_t u_end = static_cast<uint64_t>(end);
        
        while (i < u_end) {
            svbool_t pg = svwhilelt_b32(i, u_end);
            svfloat32_t v_dst = svld1_f32(pg, dst + i);
            svfloat32_t v_src = svld1_f32(pg, src + i);
            svfloat32_t v_res = svadd_f32_z(pg, v_dst, v_src); // 原生 FP32 加法
            svst1_f32(pg, dst + i, v_res);
            i += vl;
        }
    });
}


template <>
void sve_reduce_kernel_impl<at::Half>(at::Half* dst, const at::Half* src, size_t num_elements) {
    int64_t grain_size = get_optimal_grain_size(num_elements, sizeof(at::Half));

    at::parallel_for(0, num_elements, grain_size, [&](int64_t start, int64_t end) {
        uint64_t vl = svcnth(); // Count Halfwords (16位)
        uint64_t i = static_cast<uint64_t>(start);
        uint64_t u_end = static_cast<uint64_t>(end);
        
        // 必须强转成 SVE 认识的 float16_t 指针
        float16_t* dst_f16 = reinterpret_cast<float16_t*>(dst);
        const float16_t* src_f16 = reinterpret_cast<const float16_t*>(src);

        while (i < u_end) {
            svbool_t pg = svwhilelt_b16(i, u_end);
            svfloat16_t v_dst = svld1_f16(pg, dst_f16 + i);
            svfloat16_t v_src = svld1_f16(pg, src_f16 + i);
            svfloat16_t v_res = svadd_f16_z(pg, v_dst, v_src); // 原生 FP16 加法
            svst1_f16(pg, dst_f16 + i, v_res);
            i += vl;
        }
    });
}


template <>
void sve_reduce_kernel_impl<int8_t>(int8_t* dst, const int8_t* src, size_t num_elements) {
    
    int64_t grain_size = get_optimal_grain_size(num_elements, sizeof(int8_t));
    
    at::parallel_for(0, num_elements, grain_size, [&](int64_t start, int64_t end) {
        uint64_t vl = svcntb(); // Count Bytes (8位)
        uint64_t i = static_cast<uint64_t>(start);
        uint64_t u_end = static_cast<uint64_t>(end);

        while (i < u_end) {
            svbool_t pg = svwhilelt_b8(i, u_end); // 8位断言
            svint8_t v_dst = svld1_s8(pg, dst + i);
            svint8_t v_src = svld1_s8(pg, src + i);
            svint8_t v_res = svadd_s8_z(pg, v_dst, v_src); // 原生 INT8 加法
            svst1_s8(pg, dst + i, v_res);
            i += vl;
        }
    });
}


template <>
void sve_reduce_kernel_impl<at::BFloat16>(at::BFloat16* dst, const at::BFloat16* src, size_t num_elements) {
    
    int64_t grain_size = get_optimal_grain_size(num_elements, sizeof(at::BFloat16));

    at::parallel_for(0, num_elements, grain_size, [&](int64_t start, int64_t end) {
        uint64_t vl = svcnth(); 
        uint64_t i = static_cast<uint64_t>(start);
        uint64_t u_end = static_cast<uint64_t>(end);
        
        uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(dst);
        const uint16_t* src_ptr = reinterpret_cast<const uint16_t*>(src);

        while (i < u_end) {
            svbool_t pg = svwhilelt_b16(i, u_end);
            svuint16_t v_dst = svld1_u16(pg, dst_ptr + i);
            svuint16_t v_src = svld1_u16(pg, src_ptr + i);

            svbool_t pg_lo = svunpklo_b(pg);
            svuint32_t dst_u32_lo = svlsl_n_u32_z(pg_lo, svunpklo_u32(v_dst), 16);
            svuint32_t src_u32_lo = svlsl_n_u32_z(pg_lo, svunpklo_u32(v_src), 16);
            svfloat32_t f_res_lo = svadd_f32_z(pg_lo, svreinterpret_f32_u32(dst_u32_lo), svreinterpret_f32_u32(src_u32_lo));

            svbool_t pg_hi = svunpkhi_b(pg);
            svuint32_t dst_u32_hi = svlsl_n_u32_z(pg_hi, svunpkhi_u32(v_dst), 16);
            svuint32_t src_u32_hi = svlsl_n_u32_z(pg_hi, svunpkhi_u32(v_src), 16);
            svfloat32_t f_res_hi = svadd_f32_z(pg_hi, svreinterpret_f32_u32(dst_u32_hi), svreinterpret_f32_u32(src_u32_hi));

            svuint32_t res_u32_lo = svlsr_n_u32_z(pg_lo, svreinterpret_u32_f32(f_res_lo), 16);
            svuint32_t res_u32_hi = svlsr_n_u32_z(pg_hi, svreinterpret_u32_f32(f_res_hi), 16);

            svuint16_t v_res = svuzp1_u16(svreinterpret_u16_u32(res_u32_lo), svreinterpret_u16_u32(res_u32_hi));
            svst1_u16(pg, dst_ptr + i, v_res);
            
            i += vl;
        }
    });
}


void sve_reduce_torch(at::Tensor dst, at::Tensor src) {
    TORCH_CHECK(dst.numel() == src.numel(), "Tensors must have the same number of elements");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(), "Tensors must have the same dtype");
    TORCH_CHECK(dst.device().is_cpu() && src.device().is_cpu(), "Tensors must be on CPU");
    TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "Tensors MUST be contiguous in memory!");

    auto dtype = dst.scalar_type();
    
    if (dtype == at::ScalarType::Float) {
        sve_reduce_kernel_impl<float>(dst.data_ptr<float>(), src.data_ptr<float>(), dst.numel());
    } else if (dtype == at::ScalarType::Half) {
        sve_reduce_kernel_impl<at::Half>(dst.data_ptr<at::Half>(), src.data_ptr<at::Half>(), dst.numel());
    } else if (dtype == at::ScalarType::BFloat16) {
        sve_reduce_kernel_impl<at::BFloat16>(dst.data_ptr<at::BFloat16>(), src.data_ptr<at::BFloat16>(), dst.numel());
    } else if (dtype == at::ScalarType::Char) { // Char 对应 PyTorch 里的 int8
        sve_reduce_kernel_impl<int8_t>(dst.data_ptr<int8_t>(), src.data_ptr<int8_t>(), dst.numel());
    } else {
        TORCH_CHECK(false, "Kunpeng SVE Reduce Unsupported dtype! Only Float32, FP16, BF16, INT8 are supported.");
    }
}

TORCH_LIBRARY_FRAGMENT(vllm, m) {
    m.def("kunpeng_all_reduce(Tensor dst, Tensor src) -> ()");
    m.impl("kunpeng_all_reduce", c10::kCPU, &sve_reduce_torch);
}